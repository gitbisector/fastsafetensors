# SPDX-License-Identifier: Apache-2.0
"""Tests for read-time tensor sharding (SafeTensorsMetadata.with_slices)."""

import pytest

from fastsafetensors import SafeTensorsMetadata

# ---- with_slices frame math ----


def _meta(input_files, framework):
    return SafeTensorsMetadata.from_file(input_files[0], framework)


def _pick_2d(meta, min_rows=2):
    for name, f in meta.tensors.items():
        if len(f.shape) == 2 and f.shape[0] >= min_rows:
            return name, f
    pytest.skip("fixture has no 2D tensor")


def test_with_slices_even_split(input_files, framework):
    meta = _meta(input_files, framework)
    name, frame = _pick_2d(meta)
    rows = frame.shape[0]
    world = 2
    parts = []
    for rank in range(world):
        d = meta.with_slices(lambda n: (0, rank, world) if n == name else None)
        nf = d.tensors[name]
        assert nf.shape[1:] == frame.shape[1:]
        parts.append(nf)
        # other tensors untouched (same object)
        for other, of in d.tensors.items():
            if other != name:
                assert of is meta.tensors[other]
    # rows partition exactly, byte ranges are adjacent and cover the original
    assert sum(p.shape[0] for p in parts) == rows
    assert parts[0].data_offsets[0] == frame.data_offsets[0]
    assert parts[-1].data_offsets[1] == frame.data_offsets[1]
    assert parts[0].data_offsets[1] == parts[1].data_offsets[0]


def test_with_slices_remainder_to_low_ranks(input_files, framework):
    meta = _meta(input_files, framework)
    name, frame = _pick_2d(meta, min_rows=3)
    rows = frame.shape[0]
    world = 3
    if rows % world == 0:
        world = rows - 1 if rows > 2 else 2  # force a remainder
    sizes = []
    for rank in range(world):
        d = meta.with_slices(lambda n: (0, rank, world) if n == name else None)
        sizes.append(d.tensors[name].shape[0])
    assert sum(sizes) == rows
    assert max(sizes) - min(sizes) <= 1
    assert sizes == sorted(sizes, reverse=True)  # remainder to lowest ranks


def test_with_slices_passthrough_cases(input_files, framework):
    meta = _meta(input_files, framework)
    name, frame = _pick_2d(meta)
    # dim != 0: not narrowable at read time
    d = meta.with_slices(lambda n: (1, 0, 2) if n == name else None)
    assert d.tensors[name] is meta.tensors[name]
    # world <= 1
    d = meta.with_slices(lambda n: (0, 0, 1) if n == name else None)
    assert d.tensors[name] is meta.tensors[name]
    # more ranks than rows
    d = meta.with_slices(lambda n: (0, 0, frame.shape[0] + 1) if n == name else None)
    assert d.tensors[name] is meta.tensors[name]
    # spec returning None for everything: identical frames
    d = meta.with_slices(lambda n: None)
    assert all(d.tensors[k] is meta.tensors[k] for k in meta.tensors)
