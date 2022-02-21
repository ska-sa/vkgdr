/*******************************************************************************
 * Copyright (c) 2022, National Research Foundation (SARAO)
 *
 * Licensed under the BSD 3-Clause License (the "License"); you may not use
 * this file except in compliance with the License. You may obtain a copy
 * of the License at
 *
 *   https://opensource.org/licenses/BSD-3-Clause
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

/**
 * @file
 */

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

/**
 * Valid flags to pass to @ref vkgdr_open.
 */
typedef enum vkgdr_open_flags
{
    /// Ignore the given device and use the current CUDA context instead
    VKGDR_OPEN_CURRENT_CONTEXT_BIT = 1,
    /// Treat the memory as non-coherent even if it is coherent (for debugging only)
    VKGDR_OPEN_FORCE_NON_COHERENT_BIT = 2,
    /// Fail unless memory is guaranteed to be coherent
    VKGDR_OPEN_REQUIRE_COHERENT_BIT = 4
} vkgdr_open_flags;

/**
 * Get the error (if any) from the previous call to @ref vkgdr_open or
 * @ref vkgdr_memory_alloc in the current thread.  If the last call was
 * successful, returns @c NULL. The caller is responsible for freeing the
 * return value.
 */
VKGDR_API char *vkgdr_last_error(void);

/**
 * Obtain a handle to the library.
 *
 * @param device   CUDA device from which to allocate the memory (ignored if
 *                 @a flags contains @ref VKGDR_OPEN_CURRENT_CONTEXT_BIT)
 * @param flags    A bitwise combination of zero or more flags from
 *                 @ref vkgdr_open_flags.
 *
 * @returns a handle, or @c NULL on error
 *
 * @see vkgdr_last_error, vkgdr_close
 */
VKGDR_API vkgdr_t vkgdr_open(CUdevice device, uint32_t flags);

/**
 * Close a handle obtained from @ref vkgdr_open.
 *
 * This must only be called after all memory allocations have been freed.
 */
VKGDR_API void vkgdr_close(vkgdr_t g);

/**
 * Allocate memory. The current CUDA context must be one associated with the
 * device passed to @ref vkgdr_open.
 *
 * @param g     A handle obtained from @ref vkgdr_open
 * @param size  Number of bytes to allocate
 * @param flags Flags for future expansion; must be 0
 *
 * @returns A handle to the memory allocation, or @c NULL on failure.
 *
 * @see vkgdr_last_error, vkgdr_memory_free
 */
VKGDR_API vkgdr_memory_t vkgdr_memory_alloc(vkgdr_t g, size_t size, uint32_t flags);

/**
 * Free memory allocated with @ref vkgdr_memory_alloc. This must be called
 * with the same CUDA context current as when @ref vkgdr_memory_alloc was
 * called.
 */
VKGDR_API void vkgdr_memory_free(vkgdr_memory_t mem);

/// Retrieve the host pointer for a memory allocation.
VKGDR_API void *vkgdr_memory_get_host_ptr(vkgdr_memory_t mem);
/// Retrieve the CUDA device pointer for a memory allocation.
VKGDR_API CUdeviceptr vkgdr_memory_get_device_ptr(vkgdr_memory_t mem);
/// Retrieve the size of a memory allocation.
VKGDR_API size_t vkgdr_memory_get_size(vkgdr_memory_t mem);
/// Determine whether a memory allocation is coherent from the host's point of view.
VKGDR_API bool vkgdr_memory_is_coherent(vkgdr_memory_t mem);
/**
 * Get the alignment requirement for flush and invalidate operations.
 * If the memory is coherent, returns 1.
 */
VKGDR_API size_t vkgdr_memory_non_coherent_atom_size(vkgdr_memory_t mem);

/**
 * Flush host writes so that they are visible to the device.
 *
 * @param offset Offset in bytes to the first byte to flush.
 * @param size   Size in bytes of the region to flush.
 *
 * @warning @a offset and @a size must be multiples of the value returned by
 * @ref vkgdr_memory_non_coherent_atom_size (except where @a offset + @a size
 * corresponds to the end of the memory allocation). Failing to observer this
 * has undefined behaviour.
 */
VKGDR_API void vkgdr_memory_flush(vkgdr_memory_t mem, size_t offset, size_t size);

/**
 * Invalidate host view of device memory so that previous device writes are
 * visible to the host.
 *
 * @param offset Offset in bytes to the first byte to invalidate.
 * @param size   Size in bytes of the region to invalidate.
 *
 * @warning @a offset and @a size must be multiples of the value returned by
 * @ref vkgdr_memory_non_coherent_atom_size (except where @a offset + @a size
 * corresponds to the end of the memory allocation). Failing to observer this
 * has undefined behaviour.
 */
VKGDR_API void vkgdr_memory_invalidate(vkgdr_memory_t mem, size_t offset, size_t size);

#ifdef __cplusplus
}
#endif

#endif /* VKGDR_H */
