from cffi import FFI

ffibuilder = FFI()
ffibuilder.cdef(r"""
    typedef ... *vkgdr_t;
    typedef ... *vkgdr_memory_t;
    typedef int... CUdevice;
    typedef int... CUdeviceptr;

    typedef enum vkgdr_open_flags
    {
        VKGDR_OPEN_CURRENT_CONTEXT_BIT,
        VKGDR_OPEN_FORCE_NON_COHERENT_BIT,
        ...
    } vkgdr_open_flags;

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
""")

ffibuilder.set_source(
    "vkgdr._vkgdr",
    '#include "vkgdr.h"',
    include_dirs=[".", "/usr/local/cuda/include"],
    sources=["vkgdr.c"],
    libraries=["dl"]
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
