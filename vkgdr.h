#ifndef VKGDR_H
#define VKGDR_H

#include <cuda.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

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
} vkgdr_open_flags;

vkgdr_t vkgdr_open(CUdevice device, uint32_t flags);
void vkgdr_close(vkgdr_t g);

vkgdr_memory_t vkgdr_memory_alloc(vkgdr_t g, size_t size, uint32_t flags);
void vkgdr_memory_free(vkgdr_memory_t mem);

void *vkgdr_memory_get_host(vkgdr_memory_t mem);
CUdeviceptr vkgdr_memory_get_device(vkgdr_memory_t mem);
size_t vkgdr_memory_get_size(vkgdr_memory_t mem);
bool vkgdr_memory_is_coherent(vkgdr_memory_t mem);
size_t vkgdr_memory_non_coherent_atom_size(vkgdr_memory_t mem);

// TODO decide how to handle unaligned offset/size
void vkgdr_memory_flush(vkgdr_memory_t mem, size_t offset, size_t size);
void vkgdr_memory_invalidate(vkgdr_memory_t mem, size_t offset, size_t size);

#ifdef __cplusplus
}
#endif

#endif /* VKGDR_H */
