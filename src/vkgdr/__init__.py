import pycuda
from ._vkgdr import ffi, lib
from ._vkgdr.lib import VKGDR_OPEN_CURRENT_CONTEXT_BIT, VKGDR_OPEN_FORCE_NON_COHERENT_BIT


# BEGIN VERSION CHECK
# Get package version when locally imported from repo or via -e develop install
try:
    import katversion as _katversion
except ImportError:
    import time as _time
    __version__ = "0.0+unknown.{}".format(_time.strftime('%Y%m%d%H%M'))
else:
    __version__ = _katversion.get_version(__path__[0])    # type: ignore
# END VERSION CHECK


class VkgdrError(RuntimeError):
    pass


class Vkgdr:
    def __init__(self, device: int, flags: int = 0) -> None:
        handle = _vkgdr.lib.vkgdr_open(device, flags)
        if not handle:
            raise VkgdrError("vkgdr_open failed")
        self._handle = _vkgdr.ffi.gc(handle, _vkgdr.lib.vkgdr_close)

    @classmethod
    def open_current_context(cls, flags: int = 0) -> None:
        return cls(0, flags | VKGDR_OPEN_CURRENT_CONTEXT_BIT)


class Memory(pycuda.driver.PointerHolderBase):
    def __init__(self, owner: Vkgdr, size: int, flags: int = 0) -> None:
        super().__init__()
        self._owner_handle = owner._handle  # Keeps it alive
        self._context = pycuda.driver.Context.get_current()  # Keeps it alive
        handle = _vkgdr.lib.vkgdr_memory_alloc(owner._handle, size, flags)
        if not handle:
            raise VkgdrError("vkgdr_memory_alloc failed")
        self._handle = handle
        host_ptr = _vkgdr.lib.vkgdr_memory_get_host(self._handle)
        self.__array_interface__ = dict(
            shape=(size,),
            typestr="|V1",
            data=(int(_vkgdr.ffi.cast("uintptr_t", host_ptr)), False),  # False means R/W
            version=3
        )

    def __del__(self, _free=_vkgdr.lib.vkgdr_memory_free) -> None:
        # Would be cleaner to use ffi.gc on the handle, but then there is
        # nothing to guarantee that our _handle will be GCed before the
        # owner's handle once the wrapper objects are GCed.
        self._context.push()
        try:
            _free(self._handle)
        finally:
            # A static method, but the class name might be inaccessible
            # when the interpreter is shutting down.
            self._context.pop()

    def get_pointer(self) -> int:
        return _vkgdr.lib.vkgdr_memory_get_device(self._handle)

    def __len__(self) -> int:
        return _vkgdr.lib.vkgdr_memory_get_size(self._handle)

    @property
    def is_coherent(self) -> bool:
        return _vkgdr.lib.vkgdr_memory_is_coherent(self._handle)

    @property
    def non_coherent_atom_size(self) -> int:
        return _vkgdr.lib.vkgdr_memory_non_coherent_atom_size(self._handle)

    def flush(self, offset: int, size: int) -> None:
        _vkgdr.lib.vkgdr_memory_flush(self._handle, offset, size)

    def invalidate(self, offset: int, size: int) -> None:
        _vkgdr.lib.vkgdr_memory_invalidate(self._handle, offset, size)
