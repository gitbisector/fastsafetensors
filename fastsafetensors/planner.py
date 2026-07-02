# SPDX-License-Identifier: Apache-2.0

"""Static fit planner: per-file chunk budgets from a device memory budget.

Given the byte sizes of every (kept) tensor -- all known upfront from
safetensors headers -- and a total ``device_memory_budget`` the load may
occupy at peak, compute a per-file chunk budget so that

    resident_bytes + live_transient_buffers <= budget

holds at every moment of the load. Budgets are large while cumulative
resident bytes are small (whole-file loads, no chunking overhead) and
decline only as the device fills, so chunking cost is paid only where the
fit actually requires it. The plan is precomputed and deterministic: same
files, same filter, same budget -> same plan. There is no runtime feedback.

Why per-file budgets are safe (the bound): while any chunk of file ``i`` is
alive, resident bytes are at most ``R[i+1]`` (only tensors of files ``<= i``
have been materialized; file i's kept bytes are charged up front) and every
live transient buffer belongs to file ``i`` or a later file ``j > i``. The
per-file budget ``B`` declines monotonically with ``R``, so every live
buffer span is ``<= B[i]``. With at most ``depth`` buffers alive,

    peak <= R[i+1] + depth * B[i] <= budget      (by choice of B[i]).
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from .common import SafeTensorsMetadata

GiB = 1024**3


class BudgetInfeasibleError(ValueError):
    """The model cannot be loaded within the given device memory budget."""


@dataclass(frozen=True)
class FileWeightStats:
    """Per-file byte accounting for the fit plan (kept tensors only)."""

    path: str
    kept_bytes: int  # sum of kept tensor bytes: resident growth
    span_bytes: int  # last kept byte - first kept byte: single-chunk buffer size
    largest_tensor: int  # chunk floor: a tensor is the atomic load unit


def pipeline_depth(queue_size: int) -> int:
    """Worst-case number of concurrently live device buffers for a queue size.

    queue_size -1: fully serial, 1 buffer. 0: unbuffered pipeline, the producer
    materializes the next buffer while the consumer holds one -> 2. n > 0: n in
    the queue + 1 being produced + 1 being consumed -> n + 2.
    """
    if queue_size < 0:
        return 1
    return queue_size + 2


def resolve_auto_budget(free_bytes: int) -> int:
    """Budget from a free-device-memory reading: keep a reserve of
    max(5% of free, 1 GiB) for allocator rounding and small runtime
    allocations. Explicit integer budgets bypass this."""
    return max(0, free_bytes - max(free_bytes // 20, GiB))


def collect_file_stats(
    metas: List[Tuple[str, SafeTensorsMetadata]],
    keep_tensor: Optional[Callable[[str], bool]] = None,
) -> List[FileWeightStats]:
    """Byte accounting per file from already-parsed headers."""
    stats = []
    for path, meta in metas:
        kept = span_start = span_end = largest = 0
        first = True
        for name, frame in meta.tensors.items():
            if keep_tensor is not None and not keep_tensor(name):
                continue
            s, e = frame.data_offsets[0], frame.data_offsets[1]
            kept += e - s
            largest = max(largest, e - s)
            if first:
                span_start, first = s, False
            span_end = max(span_end, e)
        stats.append(
            FileWeightStats(path, kept, span_end - span_start if kept else 0, largest)
        )
    return stats


def plan_file_budgets(
    stats: List[FileWeightStats],
    device_memory_budget: int,
    depth: int,
    max_batch_bytes: Optional[int] = None,
    accumulate_resident: bool = True,
    transient_multiplier: int = 1,
) -> List[int]:
    """Per-file chunk budgets satisfying the peak-memory bound.

    ``accumulate_resident=True`` models consumers that keep every yielded
    tensor (resident grows by cumulative kept bytes). ``False`` models
    consumers whose destination memory is already allocated before the load
    (e.g. copying into preallocated model parameters): resident growth is 0
    and the plan degenerates to a uniform budget of ``budget / depth``.

    ``transient_multiplier`` scales the per-buffer transient cost. On the
    O_DIRECT reader path each in-flight chunk costs ~1x its span plus a small
    fixed thread-pool (measured on GB10 unified memory: +~150 MB regardless of
    chunk size, covered by the auto-budget reserve), so 1 is correct even on
    unified-memory systems. The mmap+pin_memory fallback additionally pins the
    chunk's file pages for the copy's lifetime -- on unified memory that draws
    from the same physical pool as the device buffer, so each in-flight chunk
    costs ~2x its span: pass 2 when the fallback will be used (see
    ``unified.chunk_transient_multiplier``).

    Returns one budget per file; feed each to
    ``SafeTensorsMetadata.plan_chunks``. A budget >= the file's span yields a
    single whole-span chunk (no splitting). Raises ``BudgetInfeasibleError``
    at plan time when some file's largest tensor cannot fit.
    """
    if device_memory_budget <= 0:
        raise BudgetInfeasibleError(
            f"device_memory_budget={device_memory_budget} must be positive"
        )
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")
    if transient_multiplier < 1:
        raise ValueError(
            f"transient_multiplier must be >= 1, got {transient_multiplier}"
        )
    eff_depth = depth * transient_multiplier
    budgets = []
    resident = 0
    for st in stats:
        if accumulate_resident:
            resident += st.kept_bytes
        b = (device_memory_budget - resident) // eff_depth
        if max_batch_bytes is not None:
            b = min(b, max_batch_bytes)
        if b < st.largest_tensor:
            required = resident + eff_depth * st.largest_tensor
            raise BudgetInfeasibleError(
                f"Model does not fit device_memory_budget: loading '{st.path}' "
                f"needs >= {required} bytes ({resident} resident + {eff_depth} x "
                f"{st.largest_tensor} transient), budget is {device_memory_budget}. "
                f"Reduce pipeline depth (queue_size), free device memory, or pass "
                f"a larger explicit budget."
            )
        budgets.append(b)
    return budgets
