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

"""CFFI build script."""

from cffi import FFI

ffibuilder = FFI()
ffibuilder.cdef(
    r"""
    typedef ... *vkgdr_t;
    typedef ... *vkgdr_memory_t;
    typedef int... CUdevice;
    typedef int... CUdeviceptr;

    typedef enum vkgdr_open_flags
    {
        VKGDR_OPEN_CURRENT_CONTEXT_BIT,
        VKGDR_OPEN_FORCE_NON_COHERENT_BIT,
        VKGDR_OPEN_REQUIRE_COHERENT_BIT,
        ...
    } vkgdr_open_flags;

    char *vkgdr_last_error(void);

    vkgdr_t vkgdr_open(CUdevice device, uint32_t flags);
    void vkgdr_close(vkgdr_t g);

    vkgdr_memory_t vkgdr_memory_alloc(vkgdr_t g, size_t size, uint32_t flags);
    void vkgdr_memory_free(vkgdr_memory_t mem);

    void *vkgdr_memory_get_host_ptr(vkgdr_memory_t mem);
    CUdeviceptr vkgdr_memory_get_device_ptr(vkgdr_memory_t mem);
    size_t vkgdr_memory_get_size(vkgdr_memory_t mem);
    bool vkgdr_memory_is_coherent(vkgdr_memory_t mem);
    size_t vkgdr_memory_non_coherent_atom_size(vkgdr_memory_t mem);

    void vkgdr_memory_flush(vkgdr_memory_t mem, size_t offset, size_t size);
    void vkgdr_memory_invalidate(vkgdr_memory_t mem, size_t offset, size_t size);

    void free(void *);
"""
)

ffibuilder.set_source(
    "vkgdr._vkgdr",
    '#include "vkgdr.h"',
    include_dirs=[".", "/usr/local/cuda/include"],
    sources=["vkgdr.c"],
    libraries=["dl"],
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
