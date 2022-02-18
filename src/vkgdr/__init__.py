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

from typing import Type, TypeVar

from ._vkgdr import ffi, lib
from ._vkgdr.lib import (  # noqa: F401
    VKGDR_OPEN_CURRENT_CONTEXT_BIT,
    VKGDR_OPEN_FORCE_NON_COHERENT_BIT,
    VKGDR_OPEN_REQUIRE_COHERENT_BIT,
)

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


class VkgdrError(RuntimeError):
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
    def __init__(self, device: int, flags: int = 0) -> None:
        handle = lib.vkgdr_open(device, flags)
        if not handle:
            _raise_vkgdr_error()
        self._handle = ffi.gc(handle, lib.vkgdr_close)

    @classmethod
    def open_current_context(cls: Type[_V], flags: int = 0) -> _V:
        return cls(0, flags | VKGDR_OPEN_CURRENT_CONTEXT_BIT)


class RawMemory:
    """Low-level memory allocation.

    This should generally not be used directly, as it requires the memory
    to be explicitly freed (otherwise it will leak). Use an API-specific
    wrapper like :class:`vkgdr.pycuda.Memory` for safe garbage
    collection.
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

        This must be called with the same CUDA context active that was used
        to allocate the memory.
        """
        if self._handle:
            self._free(self._handle)
            self._handle = None
            self._owner_handle = None

    @property
    def device_ptr(self) -> int:
        return lib.vkgdr_memory_get_device_ptr(self._handle)

    def __len__(self) -> int:
        return lib.vkgdr_memory_get_size(self._handle)

    @property
    def is_coherent(self) -> bool:
        return lib.vkgdr_memory_is_coherent(self._handle)

    @property
    def non_coherent_atom_size(self) -> int:
        return lib.vkgdr_memory_non_coherent_atom_size(self._handle)

    def flush(self, offset: int, size: int) -> None:
        lib.vkgdr_memory_flush(self._handle, offset, size)

    def invalidate(self, offset: int, size: int) -> None:
        lib.vkgdr_memory_invalidate(self._handle, offset, size)
