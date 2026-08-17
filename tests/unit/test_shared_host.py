# SPDX-License-Identifier: Apache-2.0
"""Shared-host staging: every rank must see byte-identical shards.

These run real processes against real /dev/shm -- the point of the primitive is
cross-process visibility, which a threaded stand-in would not exercise.
"""

import hashlib
import multiprocessing as mp
import os

import pytest

from fastsafetensors.shared_host import (
    SharedHostRing,
    node_key,
    shared_host_available,
)

MiB = 1024 * 1024


def _make_file(path, seed, size):
    """Deterministic pseudo-random content, so mismatches are unambiguous."""
    data = hashlib.sha256(str(seed).encode()).digest()
    buf = (data * (size // len(data) + 1))[:size]
    with open(path, "wb") as f:
        f.write(buf)
    return buf


def _worker(rank, size, paths, slot_bytes, barrier, tag, out, ranges=None):
    try:
        ring = SharedHostRing(
            slot_bytes, size, rank, barrier=barrier.wait, tag=tag, pin=False
        )
        try:
            ring.publish(paths, ranges=ranges)
            # every rank reads EVERY slot, including ones it never read from disk
            digests = []
            for i in range(len(paths)):
                v = ring.slot(i)
                try:
                    digests.append(hashlib.sha256(bytes(v[:slot_bytes])).hexdigest())
                finally:
                    v.release()
            out[rank] = digests
        finally:
            ring.close()
    except Exception as e:  # surface in the parent, don't hang
        out[rank] = f"ERROR {type(e).__name__}: {e}"


def _run(size, paths, slot_bytes, tag, ranges=None):
    ctx = mp.get_context("fork")
    barrier = ctx.Barrier(size)
    mgr = ctx.Manager()
    out = mgr.dict()
    procs = [
        ctx.Process(
            target=_worker, args=(r, size, paths, slot_bytes, barrier, tag, out, ranges)
        )
        for r in range(size)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0, f"worker exited {p.exitcode}"
    return dict(out)


@pytest.mark.skipif(not shared_host_available(), reason="no writable /dev/shm")
def test_every_rank_sees_every_shard(tmp_path):
    """The whole point: a rank sees shards it never read from disk."""
    size, slot = 2, 4 * MiB
    paths, expect = [], []
    for i in range(size):
        p = str(tmp_path / f"shard{i}.bin")
        body = _make_file(p, i, 3 * MiB)
        expect.append(hashlib.sha256(body + b"\x00" * (slot - len(body))).hexdigest())
        paths.append(p)

    got = _run(size, paths, slot, tag="fst_test_all")
    assert len(got) == size, got
    for rank in range(size):
        assert got[rank] == expect, f"rank {rank} saw {got[rank]}, expected {expect}"


@pytest.mark.skipif(not shared_host_available(), reason="no writable /dev/shm")
def test_ranges_restrict_what_is_read(tmp_path):
    """With ranges, only those bytes are fetched -- the tensor_filter hook."""
    size = 2
    paths = []
    for i in range(size):
        p = str(tmp_path / f"r{i}.bin")
        _make_file(p, 100 + i, 2 * MiB)
        paths.append(p)
    # each rank reads only the first MiB of its own shard
    slot = 2 * MiB
    ranges = [[(0, MiB)] for _ in range(size)]
    got = _run(size, paths, slot, tag="fst_test_rng", ranges=ranges)

    # exactly the requested MiB is present; the rest of the slot stays zero.
    # (Digesting the whole slot is what makes the "rest stays zero" half real --
    # digesting only the head would pass even if the tail had been fetched.)
    for i, p in enumerate(paths):
        head = open(p, "rb").read(MiB)
        expect = hashlib.sha256(head + b"\x00" * (slot - MiB)).hexdigest()
        for rank in range(size):
            assert (
                got[rank][i] == expect
            ), f"rank {rank} slot {i}: restriction not honoured"
        full = hashlib.sha256(open(p, "rb").read()).hexdigest()
        assert expect != full, "fixture too small to distinguish restricted from full"


def test_validation():
    with pytest.raises(ValueError):
        SharedHostRing(0, 1, 0)
    with pytest.raises(ValueError):
        SharedHostRing(MiB, 2, 5, barrier=lambda: None)
    with pytest.raises(ValueError):
        SharedHostRing(MiB, 2, 0)  # multi-rank without a barrier


def test_node_key_is_stable():
    assert node_key() == node_key()
    assert isinstance(node_key(), int)
