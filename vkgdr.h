#ifndef VKGDR_H
#define VKGDR_H

#include <cuda.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VKGDR_API __attribute__((visibility("default")))

struct vkgdr;
typedef struct vkgdr *vkgdr_t;
struct vkgdr_memory;
typedef struct vkgdr_memory *vkgdr_memory_t;

typedef enum vkgdr_open_flags
{
    // Ignore the given device and use the current CUDA context instead
    VKGDR_OPEN_CURRENT_CONTEXT_BIT = 1,
    // Treat the memory as non-coherent even if it is coherent (for debugging only)
    VKGDR_OPEN_FORCE_NON_COHERENT_BIT = 2,
    // Support exporting memory buffers to dma-bufs
    VKGDR_OPEN_DMA_BUF_BIT = 4
} vkgdr_open_flags;

typedef enum vkgdr_memory_alloc_flags
{
    // Support exporting the memory to a dma-buf
    VKGDR_MEMORY_ALLOC_DMA_BUF_BIT = 1
} vkgdr_memory_alloc_flags;

VKGDR_API vkgdr_t vkgdr_open(CUdevice device, uint32_t flags);
VKGDR_API void vkgdr_close(vkgdr_t g);

VKGDR_API vkgdr_memory_t vkgdr_memory_alloc(vkgdr_t g, size_t size, uint32_t flags);
VKGDR_API void vkgdr_memory_free(vkgdr_memory_t mem);

VKGDR_API void *vkgdr_memory_get_host_ptr(vkgdr_memory_t mem);
VKGDR_API CUdeviceptr vkgdr_memory_get_device_ptr(vkgdr_memory_t mem);
VKGDR_API size_t vkgdr_memory_get_size(vkgdr_memory_t mem);
VKGDR_API bool vkgdr_memory_is_coherent(vkgdr_memory_t mem);
VKGDR_API size_t vkgdr_memory_non_coherent_atom_size(vkgdr_memory_t mem);

// TODO decide how to handle unaligned offset/size
VKGDR_API void vkgdr_memory_flush(vkgdr_memory_t mem, size_t offset, size_t size);
VKGDR_API void vkgdr_memory_invalidate(vkgdr_memory_t mem, size_t offset, size_t size);

VKGDR_API int vkgdr_memory_export_dma_buf(vkgdr_memory_t mem);

#ifdef __cplusplus
}
#endif

#endif /* VKGDR_H */
