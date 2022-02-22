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

"""Integration of vkgdr with pycuda."""

import pycuda.driver

import vkgdr


class Memory(pycuda.driver.PointerHolderBase, vkgdr.RawMemory):
    """Memory allocation that is automatically garbage collected.

    For the garbage collection to work, the CUDA context must not be destroyed
    before the memory is freed (if necessary, with :meth:`free`). This object
    keeps a reference to the context so you do not need to worry about it
    being garbage collected, but explicitly destroying it through the pycuda
    API is likely to lead to segmentation faults.

    It can be pass as the `gpudata` kwarg to :class:`pycuda.gpuarray.GPUArray`
    to wrap it into a GPUArray that can be used with pucuda.

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

    def __init__(self, owner: vkgdr.Vkgdr, size: int, flags: int = 0) -> None:
        # The context is needed so it can be made current to free the memory
        self._context = pycuda.driver.Context.get_current()
        if self._context is None:
            raise vkgdr.VkgdrError("no current CUDA context")
        vkgdr.RawMemory.__init__(self, owner, size, flags)
        pycuda.driver.PointerHolderBase.__init__(self)

    def get_pointer(self) -> int:  # PointerHolderBase interface  # noqa: D102
        return self.device_ptr

    def free(self) -> None:
        """Free the memory.

        It is safe to call it multiple times.
        """
        context = self._context
        if context is None:
            return  # We've already freed
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
