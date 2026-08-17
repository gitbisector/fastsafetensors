# SPDX-License-Identifier: Apache-2.0
"""Shared-host staging: co-located ranks publish shards through host memory
instead of broadcasting them across the interconnect.

PROTOTYPE / RFC -- see docs at the bottom for what is deliberately not done yet.

Motivation
----------
``ParallelLoader``'s broadcast mode already has the right *read plan*:
``_create_batches`` gives every rank a different file of each batch, so each byte
leaves storage exactly once and all ranks read in parallel. What costs is the
*delivery* -- every rank must end up with every tensor, and today that means a
cross-rank broadcast.

Measured on a 155.42 GiB checkpoint, TP=2, single node, one NVMe:

    all_local=True   (every rank reads everything) ...........  41.3 s
    broadcast        (read once, redistribute over interconnect) 336 s
    shared-host      (read once, redistribute via host memory) .. ~23 s [1]

[1] measured with an equivalent consumer-side prototype; see the RFC notes.

When the ranks are on the same node the broadcast is avoidable entirely: publish
each shard into a host buffer every rank maps, and the redistribution becomes
addressing rather than transfer. Storage is the scarce resource (this NVMe peaks
at ~9.2 GB/s; a single reader gets only ~1.96 GB/s), so reading once with N
parallel readers and paying nothing to redistribute is strictly better than either
existing mode -- on one node.

What this adds
--------------
A staging primitive, not a new reader and not a new read plan:

    ring = SharedHostRing(slot_bytes, size, rank, barrier)   # one slot per rank
    ring.publish(paths_for_this_batch)             # each rank preads its own file
    for i, path in enumerate(batch):               # every rank now sees them all
        view = ring.slot(i)

It composes with, rather than replaces, the existing machinery: the read plan
still comes from ``_create_batches``, byte-range selection still comes from
``tensor_filter``/``select_byte_ranges``, and chunking still comes from
``max_batch_bytes``/``device_memory_budget``.
"""

import ctypes
import mmap
import os
import threading
from typing import List, Optional, Sequence

__all__ = ["SharedHostRing", "node_key", "shared_host_available"]

_DEFAULT_DIR = "/dev/shm"


def node_key() -> int:
    """A stable digest of this host, for deciding whether ranks are co-located.

    Shared-host staging is only meaningful for ranks on one node. The caller
    compares this across the group (an all-gather / all-same); the process-group
    abstraction has no such primitive today -- see the RFC notes.
    """
    import hashlib

    return int.from_bytes(
        hashlib.sha256(os.uname().nodename.encode()).digest()[:8], "little"
    ) & ((1 << 62) - 1)


def shared_host_available(directory: str = _DEFAULT_DIR) -> bool:
    """True when ``directory`` is usable for cross-process shared mappings."""
    return os.path.isdir(directory) and os.access(directory, os.W_OK)


class SharedHostRing:
    """One host slot per rank, mapped by every rank of the group.

    Each slot is written by exactly one rank (its owner) and read by all of them,
    so publication needs a single barrier and no locking.
    """

    def __init__(
        self,
        slot_bytes: int,
        size: int,
        rank: int,
        barrier=None,
        directory: str = _DEFAULT_DIR,
        tag: str = "fst_shared",
        pin: bool = True,
        read_threads: int = 16,
    ):
        if slot_bytes <= 0:
            raise ValueError(f"slot_bytes must be positive, got {slot_bytes}")
        if not 0 <= rank < size:
            raise ValueError(f"rank {rank} out of range for size {size}")
        if size > 1 and barrier is None:
            raise ValueError("multi-rank staging requires a barrier callable")
        # Deliberately NOT a process group: staging needs only a barrier, and the
        # framework pg abstraction has no barrier primitive (RFC notes below).
        self._barrier_fn = barrier
        self.size = size
        self.rank = rank
        self.slot_bytes = slot_bytes
        self.read_threads = read_threads
        self._paths = [os.path.join(directory, f"{tag}.{i}") for i in range(size)]
        # The owner creates its own slot; a barrier makes them visible to all.
        with open(self._paths[self.rank], "wb") as f:
            f.truncate(slot_bytes)
        self._barrier()
        self._files = [open(p, "r+b") for p in self._paths]
        self._mm = [mmap.mmap(f.fileno(), slot_bytes) for f in self._files]
        self._pinned: List[int] = []
        self._cudart: Optional[ctypes.CDLL] = None
        if pin:
            self._pin()

    # -- lifecycle ---------------------------------------------------------
    def _pin(self) -> None:
        """Page-lock the mappings so device copies out of them can DMA.

        Best effort: an unpinned ring is slower to copy from, never incorrect.
        """
        cudart: Optional[ctypes.CDLL] = None
        try:
            cudart = ctypes.CDLL("libcudart.so")
            cudart.cudaHostRegister.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_uint,
            ]
            cudart.cudaHostUnregister.argtypes = [ctypes.c_void_p]
            for m in self._mm:
                addr = ctypes.addressof(ctypes.c_char.from_buffer(m))
                rc = cudart.cudaHostRegister(ctypes.c_void_p(addr), self.slot_bytes, 1)
                if rc != 0:
                    raise RuntimeError(f"cudaHostRegister rc={rc}")
                self._pinned.append(addr)
            self._cudart = cudart
        except Exception:
            if cudart is not None:
                for addr in self._pinned:
                    try:
                        cudart.cudaHostUnregister(ctypes.c_void_p(addr))
                    except Exception:
                        pass
            self._pinned = []
            self._cudart = None

    def close(self) -> None:
        cudart = self._cudart
        if cudart is not None:
            for addr in self._pinned:
                try:
                    cudart.cudaHostUnregister(ctypes.c_void_p(addr))
                except Exception:
                    pass
        self._pinned = []
        for m in self._mm:
            try:
                m.close()
            except Exception:
                pass
        for f in self._files:
            f.close()
        try:
            os.unlink(self._paths[self.rank])
        except OSError:
            pass

    def __enter__(self) -> "SharedHostRing":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- publication -------------------------------------------------------
    def publish(
        self,
        batch: Sequence[Optional[str]],
        ranges: Optional[Sequence[Sequence[tuple]]] = None,
    ) -> None:
        """Read this rank's file of ``batch`` into its slot; return when all have.

        ``batch[i]`` is the path rank *i* owns (``None`` if that rank has no file
        in this batch, which happens in the last batch). ``ranges[i]``, when given,
        restricts the read to those absolute byte ranges -- this is where a
        ``tensor_filter`` narrows what is fetched.
        """
        if len(batch) > self.size:
            raise ValueError(f"batch of {len(batch)} exceeds group size {self.size}")
        mine = batch[self.rank] if self.rank < len(batch) else None
        if mine is not None:
            spans = list(ranges[self.rank]) if ranges else [(0, os.path.getsize(mine))]
            _pread_into(mine, self._mm[self.rank], spans, self.read_threads)
        self._barrier()

    def slot(self, index: int) -> memoryview:
        """A read-only view of the slot owned by rank ``index``.

        Valid until the next ``publish``; callers must release views before then.
        """
        return memoryview(self._mm[index])

    def _barrier(self) -> None:
        if self.size > 1 and self._barrier_fn is not None:
            self._barrier_fn()


def _pread_into(path: str, mm, spans, threads: int) -> None:
    """Read ``spans`` of ``path`` into ``mm`` at matching offsets."""
    fd = os.open(path, os.O_RDONLY)
    try:
        CHUNK = 16 << 20
        jobs = [(a, min(a + CHUNK, e)) for a, e in spans for a in range(a, e, CHUNK)]

        def run(sub):
            for a, b in sub:
                got = 0
                while got < b - a:
                    data = os.pread(fd, min(8 << 20, b - a - got), a + got)
                    if not data:
                        break
                    mm[a + got : a + got + len(data)] = data
                    got += len(data)

        step = max(1, (len(jobs) + threads - 1) // threads)
        workers = [
            threading.Thread(target=run, args=(jobs[i : i + step],))
            for i in range(0, len(jobs), step)
        ]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# RFC notes -- what this prototype deliberately does not do yet
#
# * The pg abstraction needs a barrier(). Staging needs exactly one collective and
#   the framework process group exposes only broadcast/size/rank, so this takes an
#   injected callable. Adding barrier() to ProcessGroupBase is part of the proposal.
# * Delivery only. It stages bytes; it does not yet replace the broadcast inside
#   ParallelLoader.iterate_weights(). Wiring it in means: where a batch is
#   broadcast today, publish() instead and build each rank's frames against
#   slot(i). The read plan (_create_batches) is unchanged.
# * Budgeting. The fit planner charges DEVICE memory. Shared-host staging adds a
#   host ring (size x slot_bytes) and, for consumers that stage to device, a
#   device buffer. Both must be charged or the load can OOM mid-flight -- exactly
#   what device_memory_budget exists to turn into a plan-time error.
# * Multi-node. node_key() gates it; the general shape is shared-host WITHIN a
#   node and broadcast/allgather ACROSS nodes, which also fixes the multi-node
#   case where every node currently re-reads the whole checkpoint.
# * Unified memory. On unified-memory parts the ring IS device memory, so the
#   device-side copy disappears; this should pay more there than on discrete GPUs.
# * GDS. With a working GDS path there is no host buffer to share, so the
#   equivalent is split reads plus a peer exchange. Choose per platform, by
#   measurement, never by default.
# ---------------------------------------------------------------------------
