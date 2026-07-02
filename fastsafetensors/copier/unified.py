# SPDX-License-Identifier: Apache-2.0

"""Unified memory copier for systems with shared CPU/GPU memory (DGX Spark, Grace Hopper).

Uses mmap → pin_memory → cudaMemcpyAsync instead of the bounce buffer approach.
On unified memory with ATS, pin_memory on mmap'd pages triggers kernel readahead
and page pinning in a single optimized path, then async DMA transfers at full
memory bandwidth.

All framework-specific operations (mmap + pinning, device synchronization,
device-name detection) go through the FrameworkOpBase abstraction so this
module never imports torch or paddle directly.
"""

import os
from typing import Dict, List, Optional, Set, Tuple

from .. import cpp as fstcpp
from ..common import SafeTensorsMetadata, get_fs_type, init_logger
from ..frameworks import FrameworkOpBase, TensorBase
from ..st_types import Device, DType
from .base import CopierInterface, validated_byte_ranges
from .registry import CopierConstructFunc, register_copier_constructor

logger = init_logger(__name__)

# O_DIRECT bypasses the page cache, which wins on local block devices but
# forfeits kernel readahead / client caching on network filesystems, where the
# buffered mmap + pin path performs better. Gate the fast path by fs type;
# FASTSAFETENSORS_ODIRECT=1/0 forces it on/off regardless.
_NETWORK_FS = {
    "nfs",
    "nfs4",
    "cifs",
    "smb3",
    "smbfs",
    "sshfs",
    "fuse.sshfs",
    "lustre",
    "gpfs",
    "beegfs",
    "glusterfs",
    "ceph",
    "9p",
    "virtiofs",
}
_warned_fs: set = set()


def _odirect_ok(path: str) -> bool:
    override = os.environ.get("FASTSAFETENSORS_ODIRECT")
    if override is not None:
        return override == "1"
    fstype = get_fs_type(path)
    if fstype in _NETWORK_FS:
        if fstype not in _warned_fs:
            _warned_fs.add(fstype)
            logger.info(
                "checkpoint on network filesystem (%s): using buffered reads "
                "instead of O_DIRECT (set FASTSAFETENSORS_ODIRECT=1 to force)",
                fstype,
            )
        return False
    return True


class UnifiedMemCopier(CopierInterface):
    """Copier using mmap → pin_memory → cudaMemcpyAsync for unified memory.

    On systems where CPU and GPU share the same physical memory (DGX Spark,
    Grace Hopper), this avoids the unnecessary bounce buffer used by NoGdsFileCopier.
    The mmap + pin_memory path lets the kernel handle readahead and page pinning
    in a single step, then async DMA copies at full memory bandwidth.
    """

    def __init__(
        self,
        metadata: SafeTensorsMetadata,
        device: Device,
        framework: FrameworkOpBase,
    ):
        self.metadata = metadata
        self.device = device
        self.framework = framework
        self._pinned: List[TensorBase] = []
        self.byte_ranges: Optional[List[Tuple[int, int]]] = None
        self._chunk_names: Optional[Set[str]] = None
        self._base_off = metadata.header_length
        # Worker count for the C++ O_DIRECT range reader (dma_load_runs). Single
        # -thread pin_memory is page-cache-bound (~2.5 GB/s); O_DIRECT threads
        # bypass the cache and drive NVMe queue depth. Falls back to pin_memory
        # if the reader is unavailable.
        self._dma_threads = int(os.environ.get("FASTSAFETENSORS_DMA_THREADS", "8"))

    def set_byte_ranges(self, byte_ranges: Optional[List[Tuple[int, int]]]) -> None:
        """Restrict reads to these ``[start, end)`` absolute file-offset runs.

        Only the bytes in the given runs are mmap-faulted, pinned, and copied;
        the rest of the device buffer is left uninitialized (so the corresponding
        tensors must not be requested). Tensor offsets are unchanged. ``None``
        reads the whole data section. Build runs with
        ``SafeTensorsMetadata.select_byte_ranges``.
        """
        self.byte_ranges = validated_byte_ranges(self.metadata, byte_ranges)

    def set_chunk(self, byte_ranges: List[Tuple[int, int]], names: Set[str]) -> None:
        """Load only ``names`` into a buffer sized to those tensors' span.

        Allocates only ``max_end - min_start`` of the runs and instantiates just
        ``names``, so peak device memory is bounded by the chunk rather than the
        shard -- the building block for ``max_batch_bytes`` sub-file batching.
        """
        self.byte_ranges = byte_ranges
        self._chunk_names = names

    def submit_io(
        self, use_buf_register: bool, max_copy_block_size: int
    ) -> fstcpp.gds_device_buffer:
        header_length = self.metadata.header_length

        # Default to the whole data section, reproducing the full-file read.
        # An empty list (vs None) reads nothing — same semantics as nogds.
        runs = self.byte_ranges
        if runs is None:
            runs = [(header_length, self.metadata.size_bytes)]

        if self._chunk_names is not None:
            # Compact chunk: allocate only the runs' span and map gbuf[0] to the
            # first run's start, so peak memory tracks the chunk, not the shard.
            base_off = min(s for s, _ in runs)
            alloc_length = max(e for _, e in runs) - base_off
        else:
            base_off = header_length
            alloc_length = self.metadata.size_bytes - header_length
        self._base_off = base_off

        # Allocate CUDA buffer via framework's allocator (proper lifecycle)
        gbuf = self.framework.alloc_tensor_memory(alloc_length, self.device)

        # Fast path: multithreaded O_DIRECT reader copies only the runs straight
        # into gbuf (byte F -> gbuf[F - base_off]), bypassing the page cache and
        # single-thread pin. Works for both full and compact-chunk buffers.
        # FASTSAFETENSORS_DMA_THREADS=0 disables it (falls back to mmap + pin);
        # network filesystems fall back automatically (see _odirect_ok).
        dma_load_runs = getattr(fstcpp, "dma_load_runs", None)
        if (
            dma_load_runs is not None
            and self._dma_threads > 0
            and _odirect_ok(self.metadata.src)
        ):
            starts = [s for s, _ in runs]
            ends = [e for _, e in runs]
            rc = dma_load_runs(
                gbuf.get_base_address(),
                self.metadata.src,
                base_off,
                starts,
                ends,
                self._dma_threads,
            )
            if rc == 0:
                return gbuf
            # Non-zero: reader unusable here; fall back to mmap + pin_memory.

        base_address = gbuf.get_base_address()
        self._pinned = []
        for start, end in runs:
            # mmap_file_pinned faults in + pins only this run's pages
            # (kernel readahead + DMA-ready), then DMA to the matching offset in
            # gbuf (gbuf[0] maps to file offset base_off).
            pinned = self.framework.mmap_file_pinned(
                self.metadata.src, end - start, start
            )
            self._pinned.append(pinned)
            ret = fstcpp.memcpy_h2d_async(  # type: ignore[attr-defined]
                base_address + (start - base_off),
                pinned.data_ptr(),
                end - start,
            )
            if ret != 0:
                self.framework.free_tensor_memory(gbuf, self.device)
                self._pinned = []
                raise RuntimeError(
                    f"cudaMemcpyAsync failed with error {ret} for {self.metadata.src}"
                )

        return gbuf

    def wait_io(
        self,
        gbuf: fstcpp.gds_device_buffer,
        dtype: DType = DType.AUTO,
        noalign: bool = False,
    ) -> Dict[str, TensorBase]:
        self.framework.synchronize(self.device)

        # Alignment note: unlike the GDS copier, we only copy the data section
        # (not the header) into gbuf, so gbuf starts at a CUDA-allocator-aligned
        # address. The copy_start_offset=header_length cancels out in get_tensors'
        # pointer arithmetic, giving correct offsets. No memmove fixup needed.
        tensors = self.metadata.get_tensors(
            gbuf, self.device, self._base_off, dtype=dtype, names=self._chunk_names
        )

        # Release the pinned mmap pages
        self._pinned = []

        return tensors


def is_unified_memory_system(framework: Optional[FrameworkOpBase] = None) -> bool:
    """Detect if this system has unified CPU/GPU memory.

    Currently verified on DGX Spark (GB10). Other unified memory
    platforms (Grace Hopper GH200) may also benefit but are untested.

    Can be overridden via the FASTSAFETENSORS_UNIFIED_MEM environment
    variable: set to "1" to force enable, "0" to force disable.
    Device-name detection requires *framework*; with framework=None only
    the environment override can enable it.
    """
    override = os.environ.get("FASTSAFETENSORS_UNIFIED_MEM")
    if override is not None:
        return override == "1"

    if framework is None:
        return False
    return "gb10" in framework.get_device_name(0).lower()


@register_copier_constructor("unified")
def new_unified_copier(device: Device, **kwargs) -> CopierConstructFunc:
    """Factory function for UnifiedMemCopier.

    Returns a constructor that creates UnifiedMemCopier instances.
    """
    from .nogds import load_library_func

    load_library_func(kwargs.get("framework"))

    def construct_unified_copier(
        metadata: SafeTensorsMetadata,
        device: Device,
        framework: FrameworkOpBase,
    ) -> CopierInterface:
        return UnifiedMemCopier(metadata, device, framework)

    return construct_unified_copier
