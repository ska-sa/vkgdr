#ifndef VKGDR_H
#define VKGDR_H

#include <cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

struct vkgdr;
typedef struct vkgdr *vkgdr_t;
struct vkgdr_memory;
typedef struct vkgdr_memory *vkgdr_memory_t;

vkgdr_t vkgdr_open(CUdevice device);
vkgdr_t vkgdr_open_current(void);
void vkgdr_close(vkgdr_t g);

vkgdr_memory_t vkgdr_malloc(vkgdr_t g, size_t size);
void vkgdr_free(vkgdr_memory_t mem);

void *vkgdr_get_host(vkgdr_memory_t mem);
CUdeviceptr vkgdr_get_device(vkgdr_memory_t mem);

void vkgdr_flush(vkgdr_memory_t mem, size_t offset, size_t size);  // TODO alignment?
void vkgdr_invalidate(vkgdr_memory_t mem, size_t offset, size_t size); // TODO alignment?

#ifdef __cplusplus
}
#endif

#endif /* VKGDR_H */
