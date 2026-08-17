# SPDX-License-Identifier: Apache-2.0

"""Copier for shards already staged in host memory by another rank.

The shard's bytes are not read from storage here: a peer rank has already
published them into a host buffer this rank maps (see
``fastsafetensors.shared_host.SharedHostRing``), so the only work left is the
host->device copy. Slot byte *F* is file byte *F*, so this copier uses exactly
the absolute offsets every other copier uses, and the tensors it hands back are
built by the same ``SafeTensorsMetadata._get_tensors`` call.

Not registered as a ``copier_type``: it is selected per-file by the loader when
a staged source address is registered for that file, and never by config.
"""

import ctypes
from typing import Dict, List, Optional, Set, Tuple

from .. import cpp as fstcpp
from ..common import SafeTensorsMetadata
from ..frameworks import FrameworkOpBase, TensorBase
from ..st_types import Device, DeviceType, DType
from .base import CopierInterface, validated_byte_ranges


class SharedHostCopier(CopierInterface):
    """Copy a staged shard from a host address into a device buffer.

    Args:
        metadata: the shard's metadata, parsed from the real file. ``src`` still
            names the real file (so ``get_filename`` is unchanged); the bytes
            come from ``host_addr``.
        host_addr: address of byte 0 of the staged copy. The staged region must
            cover every range this copier is asked to read.
    """

    def __init__(
        self,
        metadata: SafeTensorsMetadata,
        device: Device,
        framework: FrameworkOpBase,
        host_addr: int,
    ):
        self.metadata = metadata
        self.device = device
        self.framework = framework
        self.host_addr = host_addr
        self.byte_ranges: Optional[List[Tuple[int, int]]] = None
        self._chunk_names: Optional[Set[str]] = None
        self._base_off = metadata.header_length

    def set_byte_ranges(self, byte_ranges: Optional[List[Tuple[int, int]]]) -> None:
        """Copy only these ``[start, end)`` absolute file-offset runs.

        The rest of the device buffer is left uninitialized, so the tensors it
        would hold must not be requested. ``None`` copies the whole data
        section. The runs must be a subset of what the publishing rank staged.
        """
        self.byte_ranges = validated_byte_ranges(self.metadata, byte_ranges)

    def set_chunk(self, byte_ranges: List[Tuple[int, int]], names: Set[str]) -> None:
        """Copy only ``names`` into a buffer sized to those tensors' span."""
        self.byte_ranges = byte_ranges
        self._chunk_names = names

    @classmethod
    def chunk_transient_multiplier(cls, paths: List[str]) -> int:
        """Per in-flight-chunk transient DEVICE cost, as a multiple of span: 1.

        The chunk buffer is the only device allocation; the staged copy it reads
        from is host memory and is charged separately (the ring is sized once,
        for the whole load, not per chunk).
        """
        return 1

    def submit_io(
        self, use_buf_register: bool, max_copy_block_size: int
    ) -> fstcpp.gds_device_buffer:
        header_length = self.metadata.header_length
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

        gbuf = self.framework.alloc_tensor_memory(alloc_length, self.device)
        base_address = gbuf.get_base_address()
        try:
            for start, end in runs:
                self._copy(base_address + (start - base_off), start, end - start)
        except Exception:
            self.framework.free_tensor_memory(gbuf, self.device)
            raise
        return gbuf

    def _copy(self, dst: int, start: int, length: int) -> None:
        """Copy ``length`` staged bytes at file offset ``start`` to ``dst``."""
        src = self.host_addr + start
        if self.device.type == DeviceType.CPU:
            ctypes.memmove(dst, src, length)
            return
        memcpy_h2d_async = getattr(fstcpp, "memcpy_h2d_async", None)
        if memcpy_h2d_async is None:
            raise RuntimeError(
                "shared-host staging to a non-CPU device needs "
                "fastsafetensors.cpp.memcpy_h2d_async, which this build does "
                "not provide; load with device='cpu' or rebuild the extension."
            )
        ret = memcpy_h2d_async(dst, src, length)
        if ret != 0:
            raise RuntimeError(
                f"cudaMemcpyAsync failed with error {ret} for {self.metadata.src}"
            )

    def wait_io(
        self,
        gbuf: fstcpp.gds_device_buffer,
        dtype: DType = DType.AUTO,
        noalign: bool = False,
    ) -> Dict[str, TensorBase]:
        # The CPU path copied synchronously; synchronize() is a no-op there.
        self.framework.synchronize(self.device)
        # Only the data section is copied, so gbuf starts at an allocator-aligned
        # address and copy_start_offset cancels in get_tensors' arithmetic --
        # same as the unified copier, no memmove fixup needed.
        return self.metadata._get_tensors(
            gbuf, self.device, self._base_off, dtype=dtype, names=self._chunk_names
        )
