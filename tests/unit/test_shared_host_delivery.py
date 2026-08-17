# SPDX-License-Identifier: Apache-2.0
"""ParallelLoader(shared_host=True) must deliver byte-identical tensors.

The claim under test is not "it loads" but "every rank ends up with exactly the
bytes the existing delivery path would have given it". So every assertion here
compares digests of the delivered tensor bytes against a reference produced
without staging, and compares the delivered key SET too -- a rank that quietly
dropped a shard would otherwise pass.

Real forked processes against real /dev/shm: cross-process visibility is the
whole mechanism, and a threaded stand-in would map the same pages by accident.
"""

import hashlib
import multiprocessing as mp
import os
import traceback

import pytest

from fastsafetensors import ParallelLoader
from fastsafetensors.shared_host import (
    SharedHostCapacityError,
    check_capacity,
    shared_host_available,
)

pytestmark = pytest.mark.skipif(
    not shared_host_available(), reason="no writable /dev/shm"
)


class _FakeGroup:
    """A group that only has to answer size/rank.

    Under shared_host the loader itself runs single-process and the barrier is
    injected, so this is all ParallelLoader asks of the group -- which is what
    lets the test run without a torch.distributed rendezvous.
    """

    def __init__(self, size: int, rank: int):
        self._size, self._rank = size, rank

    def size(self) -> int:
        return self._size

    def rank(self) -> int:
        return self._rank


# Device under test. Defaults to CPU so the suite stays GPU-free in CI; set
# FST_TEST_DEVICE=cuda:0 to exercise SharedHostCopier._copy's memcpy_h2d_async
# branch, which the CPU path never reaches.
_DEV = os.environ.get("FST_TEST_DEVICE", "cpu")


def _tensor_digest(t) -> str:
    # .cpu() so the same digest works for device tensors -- with FST_TEST_DEVICE=cuda:0
    # this is what compares the memcpy_h2d_async result against the reference path.
    if hasattr(t, "cpu"):
        t = t.cpu()
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()


def _make_files(framework, tmp_dir, count, tensors_per_file=6):
    """``count`` shards with disjoint keys (they share one key namespace)."""
    if framework.get_name() == "pytorch":
        import torch
        from safetensors.torch import save_file

        def arange(n, off):
            return torch.arange(n, dtype=torch.float32).reshape(n // 8, 8) + off

    elif framework.get_name() == "paddle":
        import paddle
        from safetensors.paddle import save_file

        def arange(n, off):
            return paddle.arange(n, dtype="float32").reshape([n // 8, 8]) + off

    else:
        raise Exception(f"Unknown framework: {framework.get_name()}")

    paths = []
    for f in range(count):
        tensors = {}
        for i in range(tensors_per_file):
            # Sizes differ per tensor so a chunk plan makes uneven chunks.
            n = 8 * (16 + 8 * i)
            tensors[f"shard{f}.w{i}"] = arange(n, f * 100000 + i * 1000)
        path = os.path.join(tmp_dir, f"staged_{framework.get_name()}_{f}.safetensors")
        save_file(tensors, path, metadata={"fst": "staged"})
        paths.append(path)
    return paths


def _reference(files, framework, **kwargs):
    """Digests from the unmodified single-process path (no staging at all).

    Always CPU: this runs in the parent, and initialising CUDA here would poison
    fork() for the staged ranks ("Cannot re-initialize CUDA in forked
    subprocess"). Digests are device-independent, so a CPU reference is still a
    valid byte-identity target for device-staged delivery.
    """
    loader = ParallelLoader(
        None, files, device="cpu", nogds=True, framework=framework.get_name(), **kwargs
    )
    try:
        return {k: _tensor_digest(t) for k, t in loader.iterate_weights()}
    finally:
        loader.close()


def _staged_worker(rank, size, files, barrier, tag, out, framework_name, kwargs):
    try:
        loader = ParallelLoader(
            _FakeGroup(size, rank),
            files,
            device=_DEV,
            nogds=True,
            framework=framework_name,
            shared_host=True,
            shared_host_tag=tag,
            barrier=barrier.wait,
            **kwargs,
        )
        try:
            out[rank] = {k: _tensor_digest(t) for k, t in loader.iterate_weights()}
        finally:
            loader.close()
    except Exception:  # surface in the parent instead of hanging it
        out[rank] = f"ERROR {traceback.format_exc()}"


def _slow_reader_worker(rank, size, files, barrier, tag, out, framework_name, kwargs):
    """``_staged_worker``, but ``_slow_rank`` dawdles inside its first copy.

    Reading out of the ring is what races with a peer's next publish, so the
    delay goes in the copier rather than around the loader: it widens the window
    in which this rank is demonstrably still reading a slot its owner is free to
    overwrite.
    """
    slow_rank = kwargs.pop("_slow_rank")
    delay = kwargs.pop("_delay")
    if rank == slow_rank:
        import time

        from fastsafetensors.copier import host_staged

        real_copy = host_staged.SharedHostCopier._copy
        state = {"slept": False}

        def slow_copy(self, dst, start, length):
            # Once, on the first copy of the run: batch 0's files are copied in
            # sorted order, so this is the copy that reads the peer's slot.
            if not state["slept"]:
                state["slept"] = True
                time.sleep(delay)
            real_copy(self, dst, start, length)

        host_staged.SharedHostCopier._copy = slow_copy
    _staged_worker(rank, size, files, barrier, tag, out, framework_name, kwargs)


def _run_staged(size, files, tag, framework, worker=_staged_worker, **kwargs):
    # fork() cannot carry a CUDA context, and the session fixture initialises one
    # in the parent whenever a GPU is visible. Spawn for device runs; the worker
    # args (barrier, manager dict, framework NAME) are all picklable.
    ctx = mp.get_context("fork" if _DEV == "cpu" else "spawn")
    barrier = ctx.Barrier(size)
    mgr = ctx.Manager()
    out = mgr.dict()
    procs = [
        ctx.Process(
            target=worker,
            args=(r, size, files, barrier, tag, out, framework.get_name(), kwargs),
        )
        for r in range(size)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
        assert p.exitcode == 0, f"worker exited {p.exitcode}"
    got = dict(out)
    assert len(got) == size, f"only {sorted(got)} reported"
    for rank, val in got.items():
        assert isinstance(val, dict), f"rank {rank}: {val}"
    return got


def _assert_identical(got, expected, size):
    for rank in range(size):
        assert set(got[rank]) == set(
            expected
        ), f"rank {rank} delivered a different key set"
        for key, digest in expected.items():
            assert got[rank][key] == digest, f"rank {rank}: {key} differs byte-wise"


def _tag(name):
    # Unique per test run so concurrent suites cannot share slot files.
    return f"fst_dlv_{os.getpid()}_{name}"


def test_staged_delivery_is_byte_identical(tmp_dir, framework):
    """Two ranks, two shards: each reads one, both must end up with both."""
    size = 2
    files = _make_files(framework, tmp_dir, count=size)
    expected = _reference(files, framework)
    got = _run_staged(size, files, _tag("basic"), framework)
    _assert_identical(got, expected, size)
    # The fixture must be able to tell shards apart, or "identical" is vacuous.
    assert len({d for d in expected.values()}) == len(expected)


def test_staged_delivery_multiple_batches(tmp_dir, framework):
    """More shards than ranks: the ring is reused, so slots are overwritten."""
    size = 2
    files = _make_files(framework, tmp_dir, count=5)  # 3 batches, last one short
    expected = _reference(files, framework)
    got = _run_staged(size, files, _tag("multi"), framework)
    _assert_identical(got, expected, size)


def test_staged_delivery_with_chunk_plan(tmp_dir, framework):
    """max_batch_bytes: several chunk-batches per shard, partial staging."""
    size = 2
    files = _make_files(framework, tmp_dir, count=size, tensors_per_file=8)
    # Small enough to split every shard into several chunks, large enough for
    # the biggest tensor (8*(16+8*7)=576 floats = 2304 bytes).
    kwargs = {"max_batch_bytes": 4096}
    expected = _reference(files, framework, **kwargs)
    got = _run_staged(size, files, _tag("chunk"), framework, **kwargs)
    _assert_identical(got, expected, size)


def test_staged_delivery_deep_queue(tmp_dir, framework):
    """queue_size > 0: batches overlap, so the pre-publish barrier matters."""
    size = 2
    files = _make_files(framework, tmp_dir, count=4)
    kwargs = {"queue_size": 2}
    expected = _reference(files, framework, **kwargs)
    got = _run_staged(size, files, _tag("queued"), framework, **kwargs)
    _assert_identical(got, expected, size)


def test_publish_waits_for_the_previous_batch_readers(tmp_dir, framework):
    """A rank must not refill its slot while a peer still reads the last batch.

    Slots are reused every batch, and nothing in the loader couples one rank's
    progress to another's: rank 0 can finish batch 0 and reach batch 1's publish
    while rank 1 is still copying batch 0 -- half of which lives in *rank 0's*
    slot. The pre-publish barrier in ``_publish_batch`` is what closes that
    window, and every other test here passes with it deleted, so this is the one
    that holds it in place. Delaying rank 1's first copy makes the window wide
    enough that the outcome is decided by the barrier, not by scheduling luck.
    """
    size = 2
    files = _make_files(framework, tmp_dir, count=4)  # 2 batches, so slots recycle
    expected = _reference(files, framework)
    got = _run_staged(
        size,
        files,
        _tag("racy"),
        framework,
        worker=_slow_reader_worker,
        _slow_rank=1,
        _delay=2.0,
    )
    # Without the barrier rank 1 reads batch 1's shard out of rank 0's slot, so
    # it reports a full key set with wrong bytes -- caught by digest, not by count.
    _assert_identical(got, expected, size)


def test_shared_host_is_a_noop_for_a_single_process(tmp_dir, framework):
    """Opt-in but degenerate: nothing to stage, and nothing must change."""
    files = _make_files(framework, tmp_dir, count=2)
    expected = _reference(files, framework)
    loader = ParallelLoader(
        None,
        files,
        device=_DEV,
        nogds=True,
        framework=framework.get_name(),
        shared_host=True,
    )
    try:
        assert loader.shared_host is False
        assert loader._ring is None
        got = {k: _tensor_digest(t) for k, t in loader.iterate_weights()}
    finally:
        loader.close()
    assert got == expected


def _planner_worker(rank, size, files, barrier, tag, out, framework_name, budget):
    """Record the depth the fit planner is charged, then load normally."""
    try:
        from fastsafetensors import _planner

        real = _planner.plan_file_budgets
        seen = []

        def spy(stats, device_memory_budget, depth, **kw):
            seen.append((depth, kw.get("group_size")))
            return real(stats, device_memory_budget, depth, **kw)

        _planner.plan_file_budgets = spy
        try:
            loader = ParallelLoader(
                _FakeGroup(size, rank),
                files,
                device=_DEV,
                nogds=True,
                framework=framework_name,
                shared_host=True,
                shared_host_tag=tag,
                barrier=barrier.wait,
                device_memory_budget=budget,
                queue_size=0,
            )
        finally:
            _planner.plan_file_budgets = real
        try:
            digests = {k: _tensor_digest(t) for k, t in loader.iterate_weights()}
        finally:
            loader.close()
        out[rank] = {"seen": seen, "digests": digests}
    except Exception:
        out[rank] = f"ERROR {traceback.format_exc()}"


def test_shared_host_charges_the_planner_for_the_whole_group(tmp_dir, framework):
    """Staging holds group_size chunk buffers per batch, not one plus a receive.

    Under broadcast the planner is charged pipeline_depth + 1 (the in-flight
    receive tensor). Under staging every rank materializes the whole group
    itself, so the charge is pipeline_depth x group_size -- and getting that
    wrong is an OOM mid-load, which is exactly what a budget exists to prevent.
    """
    from fastsafetensors._planner import pipeline_depth

    size = 2
    files = _make_files(framework, tmp_dir, count=size)
    expected = _reference(files, framework)

    # fork() cannot carry a CUDA context, and the session fixture initialises one
    # in the parent whenever a GPU is visible. Spawn for device runs; the worker
    # args (barrier, manager dict, framework NAME) are all picklable.
    ctx = mp.get_context("fork" if _DEV == "cpu" else "spawn")
    barrier = ctx.Barrier(size)
    mgr = ctx.Manager()
    out = mgr.dict()
    procs = [
        ctx.Process(
            target=_planner_worker,
            args=(
                r,
                size,
                files,
                barrier,
                _tag("planner"),
                out,
                framework.get_name(),
                1 << 24,
            ),
        )
        for r in range(size)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
        assert p.exitcode == 0, f"worker exited {p.exitcode}"

    broadcast_depth = pipeline_depth(0) + 1
    staged_depth = pipeline_depth(0) * size
    assert staged_depth != broadcast_depth, "test cannot distinguish the two"
    for rank in range(size):
        got = out[rank]
        assert isinstance(got, dict), f"rank {rank}: {got}"
        assert got["seen"] == [(staged_depth, size)], f"rank {rank}: {got['seen']}"
        assert got["digests"] == expected, f"rank {rank} delivered different bytes"


def test_shared_host_with_no_files_builds_no_ring(framework):
    """Nothing to stage: no ring, and no crash sizing one from an empty list."""
    loader = ParallelLoader(
        _FakeGroup(2, 0),
        [],
        device=_DEV,
        nogds=True,
        framework=framework.get_name(),
        shared_host=True,
        barrier=lambda: None,
    )
    try:
        assert loader.shared_host is False
        assert loader._ring is None
        assert list(loader.iterate_weights()) == []
    finally:
        loader.close()


def test_shared_host_rejects_incompatible_options(tmp_dir, framework):
    files = _make_files(framework, tmp_dir, count=2)
    with pytest.raises(ValueError, match="all_local"):
        ParallelLoader(
            None,
            files,
            device=_DEV,
            nogds=True,
            framework=framework.get_name(),
            shared_host=True,
            all_local=True,
        )
    with pytest.raises(ValueError, match="tensor_filter"):
        ParallelLoader(
            _FakeGroup(2, 0),
            files,
            device=_DEV,
            nogds=True,
            framework=framework.get_name(),
            shared_host=True,
            tensor_filter=lambda name: True,
            barrier=lambda: None,
        )


def test_ring_refuses_to_exceed_its_directory():
    """A ring that cannot fit must fail here, not SIGBUS mid-publish."""
    st = os.statvfs("/dev/shm")
    free = st.f_bavail * st.f_frsize
    check_capacity("/dev/shm", free)  # exactly free is allowed
    with pytest.raises(SharedHostCapacityError):
        check_capacity("/dev/shm", free + (1 << 20))
