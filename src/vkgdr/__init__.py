################################################################################
# Copyright (c) 2022, National Research Foundation (SARAO)
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy
# of the License at
#
#   https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

"""Parts of vkgdr that are independent of the CUDA bindings used."""

import enum
from typing import Type, TypeVar

from ._vkgdr import ffi, lib

_V = TypeVar("_V", bound="Vkgdr")


# BEGIN VERSION CHECK
# Get package version when locally imported from repo or via -e develop install
try:
    import katversion as _katversion
except ImportError:
    import time as _time

    __version__ = "0.0+unknown.{}".format(_time.strftime("%Y%m%d%H%M"))
else:
    __version__ = _katversion.get_version(__path__[0])  # type: ignore
# END VERSION CHECK


class OpenFlags(enum.IntEnum):
    """Valid flags to pass to :class:`Vkgdr`."""

    #: Ignore the given device and use the current CUDA context instead.
    CURRENT_CONTEXT_BIT = lib.VKGDR_OPEN_CURRENT_CONTEXT_BIT
    #: Treat the memory as non-coherent even if it is coherent (for debugging only).
    FORCE_NON_COHERENT_BIT = lib.VKGDR_OPEN_FORCE_NON_COHERENT_BIT
    #: Fail unless memory is guaranteed to be coherent.
    REQUIRE_COHERENT_BIT = lib.VKGDR_OPEN_REQUIRE_COHERENT_BIT


class VkgdrError(RuntimeError):
    """Exception raised on errors."""

    pass


def _raise_vkgdr_error():
    err = lib.vkgdr_last_error()
    if not err:
        msg = "unknown error"
    else:
        msg = ffi.string(err).decode("utf-8", errors="replace")
        lib.free(err)
    raise VkgdrError(msg)


class Vkgdr:
    """Handle to the library.

    Parameters
    ----------
    device
        CUDA device from which to allocate the memory (ignored if
        `flags` contains :data:`VKGDR_OPEN_CURRENT_CONTEXT_BIT`)
    flags
        A bitwise combination of zero or more flags from
        :class:`OpenFlags`.

    Raises
    ------
    VkgdrError
        if there was an error from the underlying C library
    """

    def __init__(self, device: int, flags: int = 0) -> None:
        handle = lib.vkgdr_open(device, flags)
        if not handle:
            _raise_vkgdr_error()
        self._handle = ffi.gc(handle, lib.vkgdr_close)

    @classmethod
    def open_current_context(cls: Type[_V], flags: int = 0) -> _V:
        """Construct an instance using the current CUDA context.

        This is a shortcut to pass :data:`OpenFlags.CURRENT_CONTEXT_BIT`
        in `flags`.

        Parameters
        ----------
        flags
            A bitwise combination of zero or more flags from
            :class:`OpenFlags`.
        """
        return cls(0, flags | OpenFlags.CURRENT_CONTEXT_BIT)


class RawMemory:
    """Low-level memory allocation.

    This should generally not be used directly, as it requires the memory
    to be explicitly freed (otherwise it will leak). Use an API-specific
    wrapper like :class:`vkgdr.pycuda.Memory` for safe garbage
    collection.

    Parameters
    ----------
    owner
        A :class:`Vkgdr` created with the same device as the current CUDA context.
    size
        Number of bytes to allocate
    flags
        Flags for future expansion; must be 0

    Raises
    ------
    VkgdrError
        if there was an error from the underlying C library
    """

    def __init__(self, owner: Vkgdr, size: int, flags: int = 0) -> None:
        self._handle: object = None  # So that __del__ won't fall apart if allocation fails
        self._owner_handle = owner._handle  # Keeps it alive
        handle = lib.vkgdr_memory_alloc(owner._handle, size, flags)
        if not handle:
            _raise_vkgdr_error()
        self._handle = handle
        # Ensure that it can be called even during interpreter shutdown, when
        # module globals might already have been cleared.
        self._free = lib.vkgdr_memory_free
        host_ptr = lib.vkgdr_memory_get_host_ptr(self._handle)
        self.__array_interface__ = dict(
            shape=(size,),
            typestr="|V1",
            data=(int(ffi.cast("uintptr_t", host_ptr)), False),  # False means R/W
            version=3,
        )

    def free(self) -> None:
        """Free the memory.

        This must be called with the same CUDA context current as when the
        object was constructed. It is safe to call it multiple times.
        """
        if self._handle:
            self._free(self._handle)
            self._handle = None
            self._owner_handle = None

    @property
    def device_ptr(self) -> int:
        """Retrieve the CUDA device pointer for a memory allocation."""
        return lib.vkgdr_memory_get_device_ptr(self._handle)

    def __len__(self) -> int:
        """Retrieve the size of the memory allocation."""
        return lib.vkgdr_memory_get_size(self._handle)

    @property
    def is_coherent(self) -> bool:
        """Determine whether the memory allocation is coherent from the host's point of view."""
        return lib.vkgdr_memory_is_coherent(self._handle)

    @property
    def non_coherent_atom_size(self) -> int:
        """Get the alignment requirement for flush and invalidate operations.

        If the memory is coherent, returns 1.
        """
        return lib.vkgdr_memory_non_coherent_atom_size(self._handle)

    def flush(self, offset: int, size: int) -> None:
        """Flush host writes so that they are visible to the device.

        .. warning::

            `offset` and `size` must be multiples of
            :attr:`non_coherent_atom_size` (except where `offset` + `size`
            corresponds to the end of the memory allocation).
            Failing to observe this has undefined behaviour.

        Parameters
        ----------
        offset
            Offset in bytes to the first byte to flush.
        size
            Size in bytes of the region to flush.
        """
        lib.vkgdr_memory_flush(self._handle, offset, size)

    def invalidate(self, offset: int, size: int) -> None:
        """Invalidate host view of device memory so that previous device writes are visible to the host.

        .. warning::

            `offset` and `size` must be multiples of
            :attr:`non_coherent_atom_size` (except where `offset` + `size`
            corresponds to the end of the memory allocation).
            Failing to observe this has undefined behaviour.

        Parameters
        ----------
        offset
            Offset in bytes to the first byte to invalidate.
        size
            Size in bytes of the region to invalidate.
        """
        lib.vkgdr_memory_invalidate(self._handle, offset, size)


def memcpy_stream(dest: object, src: object) -> None:
    """Copy memory from one buffer object to another using streaming writes.

    This may give better performance when copying data from normal memory to
    vkgdr-mapped memory by using write combining. It should not be treated as a
    general-purpose memory copy.

    It is currently only implemented for x86-64. On other architectures the
    function can still be used but will simply call ``memcpy``.
    """
    with ffi.from_buffer(dest, require_writable=True) as dest_c:
        with ffi.from_buffer(src) as src_c:
            if len(dest_c) != len(src_c):
                raise ValueError("buffers have different sizes")
            lib.vkgdr_memcpy_stream(dest_c, src_c, len(dest_c))
