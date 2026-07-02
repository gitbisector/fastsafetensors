# SPDX-License-Identifier: Apache-2.0
"""Robustness tests: filesystem-aware I/O paths and graceful degradation."""

import pytest

from fastsafetensors.common import get_fs_type

# ---- get_fs_type: longest-prefix mount matching ----

MOUNTS = """\
/dev/root / ext4 rw 0 0
nas:/vol /mnt/nfs nfs4 rw 0 0
/dev/nvme0n1p2 /data ext4 rw 0 0
nas:/models /data/remote nfs rw 0 0
tmpfs /dev/shm tmpfs rw 0 0
/dev/sdb1 /mnt/with\\040space ext4 rw 0 0
"""


@pytest.fixture()
def mounts_file(tmp_path):
    p = tmp_path / "mounts"
    p.write_text(MOUNTS)
    return str(p)


def test_get_fs_type_basic(mounts_file):
    assert get_fs_type("/mnt/nfs/model.safetensors", mounts_file) == "nfs4"
    assert get_fs_type("/data/m/x.safetensors", mounts_file) == "ext4"
    assert get_fs_type("/somewhere/else", mounts_file) == "ext4"  # root fallback


def test_get_fs_type_longest_prefix_wins(mounts_file):
    # /data is ext4 but /data/remote is an NFS mount inside it
    assert get_fs_type("/data/remote/model.safetensors", mounts_file) == "nfs"


def test_get_fs_type_escaped_mountpoint(mounts_file):
    assert get_fs_type("/mnt/with space/f.safetensors", mounts_file) == "ext4"


def test_get_fs_type_unreadable_mounts(tmp_path):
    assert get_fs_type("/data/x", str(tmp_path / "nope")) == ""


# ---- O_DIRECT gating on network filesystems ----


def test_odirect_gating(monkeypatch):
    from fastsafetensors.copier import unified

    monkeypatch.delenv("FASTSAFETENSORS_ODIRECT", raising=False)
    monkeypatch.setattr(unified, "get_fs_type", lambda p: "nfs4")
    assert unified._odirect_ok("/mnt/nfs/f") is False
    monkeypatch.setattr(unified, "get_fs_type", lambda p: "ext4")
    assert unified._odirect_ok("/data/f") is True
    monkeypatch.setattr(unified, "get_fs_type", lambda p: "")  # unknown: allow
    assert unified._odirect_ok("/x") is True


def test_odirect_env_override(monkeypatch):
    from fastsafetensors.copier import unified

    monkeypatch.setattr(unified, "get_fs_type", lambda p: "nfs4")
    monkeypatch.setenv("FASTSAFETENSORS_ODIRECT", "1")
    assert unified._odirect_ok("/mnt/nfs/f") is True  # forced on
    monkeypatch.setattr(unified, "get_fs_type", lambda p: "ext4")
    monkeypatch.setenv("FASTSAFETENSORS_ODIRECT", "0")
    assert unified._odirect_ok("/data/f") is False  # forced off


# ---- chunk plans must fail loudly on copiers without set_chunk ----


def test_chunk_plan_requires_set_chunk(input_files, framework):
    if framework.get_name() != "pytorch":
        pytest.skip("pytorch-only")
    from fastsafetensors import SafeTensorsFileLoader, SafeTensorsMetadata

    loader = SafeTensorsFileLoader(None, "cpu", nogds=True, framework="pytorch")
    loader.add_filenames({0: [input_files[0]]})
    meta = SafeTensorsMetadata.from_file(input_files[0], framework)
    (chunk,) = meta.plan_chunks(meta.size_bytes)  # single whole-span chunk
    loader.set_chunk_plan({input_files[0]: chunk})

    class _NoChunkCopier:  # e.g. gds/dstorage: no set_chunk support
        def __init__(self, *a, **k):
            pass

    loader.copier_constructor = lambda m, d, f: _NoChunkCopier()
    with pytest.raises(NotImplementedError, match="set_chunk"):
        loader.copy_files_to_device()


# ---- device_memory_budget in broadcast mode ----


class _FakePG:
    def size(self):
        return 2

    def rank(self):
        return 0


def _make_loader(framework):
    from fastsafetensors import SafeTensorsFileLoader

    return SafeTensorsFileLoader(None, "cpu", nogds=True, framework="pytorch")


def test_broadcast_explicit_budget_allowed(input_files, framework, tmp_path):
    if framework.get_name() != "pytorch":
        pytest.skip("pytorch-only")
    import shutil

    from fastsafetensors.parallel_loader import PipelineParallel

    f2 = str(tmp_path / "copy.safetensors")
    shutil.copy(input_files[0], f2)
    pp = PipelineParallel(
        _FakePG(),
        _make_loader(framework),
        [input_files[0], f2],
        queue_size=0,
        use_tqdm_on_load=False,
        device_memory_budget=1 << 30,  # explicit int: deterministic across ranks
    )
    # one file per rank per group -> chunk-batch specs of width 2
    assert pp.weight_files_batches
    assert all(len(spec) == 2 for spec in pp.weight_files_batches)


def test_broadcast_auto_budget_rejected(input_files, framework, tmp_path):
    if framework.get_name() != "pytorch":
        pytest.skip("pytorch-only")
    import shutil

    from fastsafetensors.parallel_loader import PipelineParallel

    f2 = str(tmp_path / "copy2.safetensors")
    shutil.copy(input_files[0], f2)
    with pytest.raises(NotImplementedError, match="all-reduce"):
        PipelineParallel(
            _FakePG(),
            _make_loader(framework),
            [input_files[0], f2],
            queue_size=0,
            use_tqdm_on_load=False,
            device_memory_budget="auto",
        )
