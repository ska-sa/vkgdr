"""Integration of vkgdr with pycuda."""

import pycuda.driver

import vkgdr


class Memory(pycuda.driver.PointerHolderBase, vkgdr.RawMemory):
    def __init__(self, owner: vkgdr.Vkgdr, size: int, flags: int = 0) -> None:
        # The context is needed so it can be made current to free the memory
        self._context = pycuda.driver.Context.get_current()
        if self._context is None:
            raise vkgdr.VkgdrError("no current CUDA context")
        vkgdr.RawMemory.__init__(self, owner, size, flags)
        pycuda.driver.PointerHolderBase.__init__(self)

    def get_pointer(self) -> int:  # PointerHolderBase interface
        return self.device_ptr

    def free(self) -> None:
        context = self._context
        if context is None:
            return   # We've already freed
        context.push()
        try:
            super().free()
            self._context = None
        finally:
            # A static method, but the class name might be inaccessible
            # when the interpreter is shutting down.
            context.pop()

    def __del__(self):
        self.free()
