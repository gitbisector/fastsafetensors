# SPDX-License-Identifier: Apache-2.0
"""Tests for the static fit planner (per-file declining chunk budgets)."""

import random

import pytest

from fastsafetensors import SafeTensorsMetadata
from fastsafetensors.planner import (
    BudgetInfeasibleError,
    FileWeightStats,
    collect_file_stats,
    pipeline_depth,
    plan_file_budgets,
    resolve_auto_budget,
)

GiB = 1024**3
MiB = 1024**2


def _st(path, kept, span=None, largest=None):
    return FileWeightStats(
        path,
        kept,
        span if span is not None else kept,
        largest if largest is not None else kept // 4 or kept,
    )


# ---- pipeline depth ----


def test_pipeline_depth_mapping():
    assert pipeline_depth(-1) == 1
    assert pipeline_depth(0) == 2
    assert pipeline_depth(1) == 3
    assert pipeline_depth(4) == 6


# ---- auto budget ----


def test_resolve_auto_budget_reserve():
    # 5% of free when that exceeds 1 GiB
    assert resolve_auto_budget(100 * GiB) == 95 * GiB
    # 1 GiB floor otherwise
    assert resolve_auto_budget(10 * GiB) == 9 * GiB
    assert resolve_auto_budget(0) == 0


# ---- planner math ----


def test_budgets_decline_monotonically():
    stats = [_st(f"f{i}", 4 * GiB) for i in range(8)]
    budgets = plan_file_budgets(stats, 40 * GiB, depth=2)
    assert budgets == sorted(budgets, reverse=True)
    # first file: (40 - 4) / 2 = 18 GiB; last: (40 - 32) / 2 = 4 GiB
    assert budgets[0] == 18 * GiB
    assert budgets[-1] == 4 * GiB


def test_whole_file_when_ample():
    # budget >> everything: every per-file budget exceeds its span, so
    # plan_chunks would return a single whole-span chunk per file.
    stats = [_st(f"f{i}", 2 * GiB) for i in range(4)]
    budgets = plan_file_budgets(stats, 100 * GiB, depth=2)
    assert all(b >= st.span_bytes for b, st in zip(budgets, stats))


def test_min_with_max_batch_bytes():
    stats = [_st("f0", 1 * GiB)]
    budgets = plan_file_budgets(stats, 100 * GiB, depth=1, max_batch_bytes=512 * MiB)
    assert budgets == [512 * MiB]


def test_accumulate_resident_false_is_uniform():
    stats = [_st(f"f{i}", 8 * GiB, largest=1 * GiB) for i in range(6)]
    budgets = plan_file_budgets(stats, 12 * GiB, depth=2, accumulate_resident=False)
    assert budgets == [6 * GiB] * 6


def test_infeasible_raises_with_details():
    # 10 files x 2 GiB resident vs 12 GiB budget: infeasible partway through
    stats = [_st(f"f{i}", 2 * GiB, largest=1 * GiB) for i in range(10)]
    with pytest.raises(BudgetInfeasibleError) as ei:
        plan_file_budgets(stats, 12 * GiB, depth=2)
    msg = str(ei.value)
    assert "does not fit" in msg and "f" in msg and "queue_size" in msg


def test_nonpositive_budget_and_bad_depth():
    with pytest.raises(BudgetInfeasibleError):
        plan_file_budgets([_st("f", 1)], 0, depth=1)
    with pytest.raises(ValueError):
        plan_file_budgets([_st("f", 1)], 1, depth=0)


def test_empty_kept_file_contributes_nothing():
    stats = [_st("f0", 4 * GiB), FileWeightStats("f1", 0, 0, 0), _st("f2", 4 * GiB)]
    budgets = plan_file_budgets(stats, 20 * GiB, depth=1)
    # empty file consumes no resident: f2's budget only reflects f0 + f2
    assert budgets[2] == 20 * GiB - 8 * GiB


def test_huge_first_file_chunks():
    stats = [
        _st("big", 20 * GiB, span=20 * GiB, largest=1 * GiB),
        _st("small", 1 * GiB),
    ]
    budgets = plan_file_budgets(stats, 30 * GiB, depth=2)
    # (30 - 20) / 2 = 5 GiB < 20 GiB span -> file will be chunked, feasible
    assert 1 * GiB <= budgets[0] < 20 * GiB


# ---- simulation property test: replay the plan, assert peak <= budget ----


def _simulate_peak(stats, budgets, depth, accumulate_resident=True):
    """Conservative replay: each file's chunks occupy <= its budget; up to
    `depth` chunk buffers live at once (a sliding window over the chunk
    sequence, oldest = being consumed). Clones happen at consumption, so
    resident covers files up to and including the OLDEST file in the window
    (its kept bytes charged up front, matching the planner's R_{i+1});
    newer produced-ahead files are transient-only."""
    chunk_seq = []  # (file_idx, transient_size)
    for i, (st, b) in enumerate(zip(stats, budgets)):
        if st.kept_bytes == 0:
            continue
        if b >= st.span_bytes:
            chunk_seq.append((i, st.span_bytes))
        else:
            n = -(-st.span_bytes // b)  # ceil: worst-case chunk count
            for _ in range(n):
                chunk_seq.append((i, b))
    kept_cumsum = []
    acc = 0
    for st in stats:
        acc += st.kept_bytes
        kept_cumsum.append(acc)  # R_{i+1}
    peak = 0
    for w in range(len(chunk_seq)):
        window = chunk_seq[max(0, w - depth + 1) : w + 1]
        oldest = window[0][0]
        resident = kept_cumsum[oldest] if accumulate_resident else 0
        peak = max(peak, resident + sum(sz for _, sz in window))
    return peak


def test_simulation_peak_within_budget():
    rng = random.Random(1234)
    for trial in range(200):
        n = rng.randint(1, 12)
        stats = []
        for i in range(n):
            kept = rng.randint(1, 64) * MiB
            largest = max(1, kept // rng.randint(2, 8))
            stats.append(FileWeightStats(f"f{i}", kept, kept, largest))
        depth = pipeline_depth(rng.choice([-1, 0, 1, 3]))
        total = sum(s.kept_bytes for s in stats)
        budget = (
            total
            + rng.randint(1, 32) * MiB
            + depth * max(s.largest_tensor for s in stats)
        )
        acc = rng.random() < 0.5
        try:
            budgets = plan_file_budgets(stats, budget, depth, accumulate_resident=acc)
        except BudgetInfeasibleError:
            continue  # planner refused: nothing to verify
        peak = _simulate_peak(stats, budgets, depth, accumulate_resident=acc)
        # acc=True: resident + transient <= budget. acc=False: the plan bounds
        # the transient side only (destinations preallocated), and the sim
        # models exactly that side -> same assertion.
        assert peak <= budget, (trial, peak, budget, acc)


# ---- collect_file_stats on real files ----


def test_collect_file_stats_matches_headers(input_files, framework):
    meta = SafeTensorsMetadata.from_file(input_files[0], framework)
    (st,) = collect_file_stats([(input_files[0], meta)])
    total = sum(f.data_offsets[1] - f.data_offsets[0] for f in meta.tensors.values())
    largest = max(f.data_offsets[1] - f.data_offsets[0] for f in meta.tensors.values())
    assert st.kept_bytes == total
    assert st.largest_tensor == largest
    assert st.span_bytes >= largest

    # filtered: keep every other tensor -> kept < total, span may include holes
    names = sorted(meta.tensors.keys())
    keep = set(names[::2])
    (fst,) = collect_file_stats([(input_files[0], meta)], lambda n: n in keep)
    assert 0 < fst.kept_bytes < total
    assert fst.span_bytes >= fst.kept_bytes


# ---- end-to-end on CPU: budgeted load is byte-identical to a plain load ----


def test_parallel_loader_device_memory_budget_cpu(input_files, framework):
    if framework.get_name() != "pytorch":
        pytest.skip("pytorch-only integration test")
    import torch
    from safetensors.torch import load_file

    from fastsafetensors import ParallelLoader

    expected = load_file(input_files[0])
    meta = SafeTensorsMetadata.from_file(input_files[0], framework)
    (st,) = collect_file_stats([(input_files[0], meta)])
    # tight budget: forces chunking (several chunks) but stays feasible
    budget = st.kept_bytes + pipeline_depth(0) * st.largest_tensor + 4096

    pl = ParallelLoader(
        pg=None,
        hf_weights_files=[input_files[0]],
        device="cpu",
        nogds=True,
        use_tqdm_on_load=False,
        device_memory_budget=budget,
    )
    got = dict(pl.iterate_weights())
    assert set(got.keys()) == set(expected.keys())
    for k in expected:
        assert torch.equal(got[k], expected[k]), k


def test_transient_multiplier_halves_budgets():
    # fallback path (mmap+pin on unified memory): each live chunk costs 2x
    # its span, so per-file budgets must halve relative to multiplier=1.
    stats = [_st("f0", 2 * GiB)]
    b1 = plan_file_budgets(stats, 100 * GiB, depth=2, transient_multiplier=1)
    b2 = plan_file_budgets(stats, 100 * GiB, depth=2, transient_multiplier=2)
    assert b2[0] * 2 == b1[0] - (b1[0] % 2)


def test_transient_multiplier_infeasible():
    # a plan feasible at 1x must fail at 2x when the budget is tight
    stats = [_st("f0", 2 * GiB)]
    budget = stats[0].kept_bytes + 2 * stats[0].largest_tensor  # depth=2, k=1 fits
    plan_file_budgets(stats, budget, depth=2, transient_multiplier=1)
    with pytest.raises(BudgetInfeasibleError):
        plan_file_budgets(stats, budget, depth=2, transient_multiplier=2)


def test_transient_multiplier_validation():
    with pytest.raises(ValueError):
        plan_file_budgets([_st("f0", GiB)], GiB, depth=1, transient_multiplier=0)
