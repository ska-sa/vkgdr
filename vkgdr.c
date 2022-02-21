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

#define VK_NO_PROTOTYPES 1  /* Prevents Vulkan headers from defining functions to link to */

#ifndef _GNU_SOURCE
# define _GNU_SOURCE
#endif

#include <unistd.h>
#include <assert.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <dlfcn.h>
#include <cuda.h>
#include <vulkan/vulkan.h>
#include <dlfcn.h>
#include "vkgdr.h"

typedef CUresult (CUDAAPI *PFN_cuCtxGetDevice)(CUdevice *);
typedef CUresult (CUDAAPI *PFN_cuDestroyExternalMemory)(CUexternalMemory);
typedef CUresult (CUDAAPI *PFN_cuDeviceGetUuid)(CUuuid *, CUdevice);
typedef CUresult (CUDAAPI *PFN_cuExternalMemoryGetMappedBuffer)(CUdeviceptr *, CUexternalMemory, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC *);
typedef CUresult (CUDAAPI *PFN_cuImportExternalMemory)(CUexternalMemory *, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC *);
// _v2 suffix is because CUDA headers internally #define cuMemFree to cuMemFree_v2 (v1 used unsigned int for device pointers)
typedef CUresult (CUDAAPI *PFN_cuMemFree_v2)(CUdeviceptr);

struct vkgdr
{
    void *libvulkan_handle;
    void *libcuda_handle;
    VkInstance instance;
    VkPhysicalDevice phys_device;
    VkDevice device;
    uint32_t memory_type;
    bool coherent;
    size_t non_coherent_atom_size;

    // The CUDA functions have a fn_ prefix to avoid being mangled by #defines in cuda.h
    PFN_cuCtxGetDevice fn_cuCtxGetDevice;
    PFN_cuDestroyExternalMemory fn_cuDestroyExternalMemory;
    PFN_cuDeviceGetUuid fn_cuDeviceGetUuid;
    PFN_cuExternalMemoryGetMappedBuffer fn_cuExternalMemoryGetMappedBuffer;
    PFN_cuImportExternalMemory fn_cuImportExternalMemory;
    PFN_cuMemFree_v2 fn_cuMemFree_v2;

    // Instance functions
    PFN_vkDestroyInstance vkDestroyInstance;
    // Device-specific functions
    PFN_vkAllocateMemory vkAllocateMemory;
    PFN_vkDestroyDevice vkDestroyDevice;
    PFN_vkFlushMappedMemoryRanges vkFlushMappedMemoryRanges;
    PFN_vkFreeMemory vkFreeMemory;
    PFN_vkGetMemoryFdKHR vkGetMemoryFdKHR;
    PFN_vkInvalidateMappedMemoryRanges vkInvalidateMappedMemoryRanges;
    PFN_vkMapMemory vkMapMemory;
    PFN_vkUnmapMemory vkUnmapMemory;
};

struct vkgdr_memory
{
    vkgdr_t owner;
    VkDeviceMemory memory;
    void *host_ptr;
    size_t size;
    CUexternalMemory ext_mem;
    CUdeviceptr device_ptr;
};

struct error_state
{
    const char *msg;  // NULL if there is no error
    VkResult vk_result;
    CUresult cu_result;
};

static __thread struct error_state last_error;

static void clear_last_error()
{
    last_error.msg = NULL;
}

static void set_generic_error(const char *msg)
{
    last_error.msg = msg;
    last_error.vk_result = VK_SUCCESS;
    last_error.cu_result = CUDA_SUCCESS;
}

static void set_vk_error(const char *msg, VkResult result)
{
    last_error.msg = msg;
    last_error.vk_result = result;
    last_error.cu_result = CUDA_SUCCESS;
}

static void set_cu_error(const char *msg, CUresult result)
{
    last_error.msg = msg;
    last_error.vk_result = VK_SUCCESS;
    last_error.cu_result = result;
}

char *vkgdr_last_error(void)
{
#define CASE(x) case x: name = #x; break
    char *out;
    if (last_error.msg == NULL)
        return NULL;
    else if (last_error.vk_result != VK_SUCCESS)
    {
        const char *name;
        switch (last_error.vk_result)
        {
            // Not a complete list of Vulkan errors, but should cover the likely ones
            CASE(VK_NOT_READY);
            CASE(VK_TIMEOUT);
            CASE(VK_ERROR_OUT_OF_HOST_MEMORY);
            CASE(VK_ERROR_OUT_OF_DEVICE_MEMORY);
            CASE(VK_ERROR_INITIALIZATION_FAILED);
            CASE(VK_ERROR_DEVICE_LOST);
            CASE(VK_ERROR_MEMORY_MAP_FAILED);
            CASE(VK_ERROR_LAYER_NOT_PRESENT);
            CASE(VK_ERROR_EXTENSION_NOT_PRESENT);
            CASE(VK_ERROR_FEATURE_NOT_PRESENT);
            CASE(VK_ERROR_INCOMPATIBLE_DRIVER);
            CASE(VK_ERROR_TOO_MANY_OBJECTS);
            // Defined in Vulkan 1.0, but missing from the CentOS 7 header file
            case -13: name = "VK_ERROR_UNKNOWN"; break;
            CASE(VK_ERROR_OUT_OF_POOL_MEMORY);
            CASE(VK_ERROR_INVALID_EXTERNAL_HANDLE);
            default: name = NULL; break;
        }
        if (name)
            asprintf(&out, "%s (%s)", last_error.msg, name);
        else
            asprintf(&out, "%s (unknown Vulkan error %d)", last_error.msg, (int) last_error.vk_result);
    }
    else if (last_error.cu_result != CUDA_SUCCESS)
    {
        const char *name;
        switch (last_error.cu_result)
        {
            /* Not a complete list, but should cover the likely cases
             * (cuGetErrorName would be better, but we don't necessarily have
             * the dynamic library open here).
             */
            CASE(CUDA_ERROR_INVALID_VALUE);
            CASE(CUDA_ERROR_OUT_OF_MEMORY);
            CASE(CUDA_ERROR_NOT_INITIALIZED);
            CASE(CUDA_ERROR_DEINITIALIZED);
            CASE(CUDA_ERROR_STUB_LIBRARY);
            CASE(CUDA_ERROR_NO_DEVICE);
            CASE(CUDA_ERROR_INVALID_DEVICE);
            CASE(CUDA_ERROR_DEVICE_NOT_LICENSED);
            CASE(CUDA_ERROR_INVALID_CONTEXT);
            CASE(CUDA_ERROR_MAP_FAILED);
            CASE(CUDA_ERROR_UNMAP_FAILED);
            CASE(CUDA_ERROR_ALREADY_MAPPED);
            CASE(CUDA_ERROR_ALREADY_ACQUIRED);
            CASE(CUDA_ERROR_NOT_MAPPED);
            CASE(CUDA_ERROR_NOT_MAPPED_AS_POINTER);
            CASE(CUDA_ERROR_ECC_UNCORRECTABLE);
            CASE(CUDA_ERROR_PEER_ACCESS_UNSUPPORTED);
            CASE(CUDA_ERROR_NVLINK_UNCORRECTABLE);
            CASE(CUDA_ERROR_FILE_NOT_FOUND);
            CASE(CUDA_ERROR_OPERATING_SYSTEM);
            CASE(CUDA_ERROR_INVALID_HANDLE);
            CASE(CUDA_ERROR_ILLEGAL_STATE);
            CASE(CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE);
            CASE(CUDA_ERROR_CONTEXT_IS_DESTROYED);
            CASE(CUDA_ERROR_TOO_MANY_PEERS);
            CASE(CUDA_ERROR_NOT_PERMITTED);
            CASE(CUDA_ERROR_NOT_SUPPORTED);
            CASE(CUDA_ERROR_SYSTEM_NOT_READY);
            CASE(CUDA_ERROR_SYSTEM_DRIVER_MISMATCH);
            CASE(CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE);
            CASE(CUDA_ERROR_TIMEOUT);
            CASE(CUDA_ERROR_EXTERNAL_DEVICE);
            CASE(CUDA_ERROR_UNKNOWN);
            default: name = NULL; break;
        }
        if (name)
            asprintf(&out, "%s (%s)", last_error.msg, name);
        else
            asprintf(&out, "%s (unknown CUDA error %d)", last_error.msg, (int) last_error.cu_result);
    }
    else
    {
        out = strdup(last_error.msg);
    }
    return out;
#undef CASE
}

vkgdr_t vkgdr_open(CUdevice device, uint32_t flags)
{
    VkResult vk_result;
    CUresult cu_result;
    clear_last_error();

    void *libvulkan_handle = dlopen("libvulkan.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (!libvulkan_handle)
    {
        set_generic_error(dlerror());
        goto fail;
    }
    void *libcuda_handle = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (!libcuda_handle)
    {
        set_generic_error(dlerror());
        goto close_libvulkan;
    }

    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr) dlsym(libvulkan_handle, "vkGetInstanceProcAddr");
#define INIT_VK_INSTANCE_PFN(instance, name) PFN_ ## name name = (PFN_ ## name) vkGetInstanceProcAddr((instance), #name)
    INIT_VK_INSTANCE_PFN(NULL, vkCreateInstance);

    vkgdr_t out;
    const VkApplicationInfo application_info =
    {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .applicationVersion = 1,
        .apiVersion = VK_API_VERSION_1_1
    };
    const VkInstanceCreateInfo instance_info =
    {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &application_info
    };
    CUuuid uuid;

    out = calloc(1, sizeof(struct vkgdr));
    out->libvulkan_handle = libvulkan_handle;
    out->libcuda_handle = libcuda_handle;
    if ((vk_result = vkCreateInstance(&instance_info, NULL, &out->instance)) != VK_SUCCESS)
    {
        set_vk_error("vkCreateInstance failed", vk_result);
        goto free_out;
    }

    INIT_VK_INSTANCE_PFN(out->instance, vkEnumeratePhysicalDevices);
    INIT_VK_INSTANCE_PFN(out->instance, vkGetPhysicalDeviceProperties2);
    INIT_VK_INSTANCE_PFN(out->instance, vkCreateDevice);
    INIT_VK_INSTANCE_PFN(out->instance, vkGetPhysicalDeviceProperties);
    INIT_VK_INSTANCE_PFN(out->instance, vkGetPhysicalDeviceMemoryProperties);
    INIT_VK_INSTANCE_PFN(out->instance, vkDestroyInstance);
    INIT_VK_INSTANCE_PFN(out->instance, vkGetDeviceProcAddr);

#define INIT_CU_PFN(out, name) (out)->fn_ ## name = (PFN_ ## name) dlsym(libcuda_handle, #name)
    INIT_CU_PFN(out, cuCtxGetDevice);
    INIT_CU_PFN(out, cuDestroyExternalMemory);
    INIT_CU_PFN(out, cuDeviceGetUuid);
    INIT_CU_PFN(out, cuExternalMemoryGetMappedBuffer);
    INIT_CU_PFN(out, cuImportExternalMemory);
    INIT_CU_PFN(out, cuMemFree_v2);

    // Find the matching Vulkan physical device
    if (flags & VKGDR_OPEN_CURRENT_CONTEXT_BIT)
    {
        if ((cu_result = out->fn_cuCtxGetDevice(&device)) != CUDA_SUCCESS)
        {
            set_cu_error("cuCtxGetDevice failed", cu_result);
            goto destroy_instance;
        }
    }
    if ((cu_result = out->fn_cuDeviceGetUuid(&uuid, device)) != CUDA_SUCCESS)
    {
        set_cu_error("cuDeviceGetUuid failed", cu_result);
        goto destroy_instance;
    }

    uint32_t n_devices;
    if ((vk_result = vkEnumeratePhysicalDevices(out->instance, &n_devices, NULL)) != VK_SUCCESS)
    {
        set_vk_error("vkEnumeratePhysicalDevices failed", vk_result);
        goto destroy_instance;
    }
    VkPhysicalDevice *devices = calloc(n_devices, sizeof(VkPhysicalDevice));
    if (devices == NULL)
    {
        set_generic_error("failed to allocate memory for device list");
        goto destroy_instance;
    }
    if ((vk_result = vkEnumeratePhysicalDevices(out->instance, &n_devices, devices)) != VK_SUCCESS)
    {
        set_vk_error("vkEnumeratePhysicalDevices failed", vk_result);
        goto free_devices;
    }
    VkPhysicalDevice phys_device = VK_NULL_HANDLE;
    for (uint32_t i = 0; i < n_devices; i++)
    {
        VkPhysicalDeviceIDProperties id_properties =
        {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES
        };
        VkPhysicalDeviceProperties2 properties2 =
        {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            .pNext = &id_properties
        };
        vkGetPhysicalDeviceProperties2(devices[i], &properties2);
        if (!memcmp(&uuid, id_properties.deviceUUID, VK_UUID_SIZE))
        {
            phys_device = devices[i];
            break;
        }
    }
    if (phys_device == VK_NULL_HANDLE)
    {
        set_generic_error("no Vulkan device matching the CUDA device found");
        goto free_devices;
    }

    const float queue_priorities[] = {1.0f};
    const VkDeviceQueueCreateInfo queue_infos[] =
    {
        {
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = 0,  // TODO: introspect the families
            .queueCount = 1,
            .pQueuePriorities = queue_priorities
        }
    };
    const char * const extensions[] = {"VK_KHR_external_memory_fd"};
    const VkDeviceCreateInfo device_info =
    {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = queue_infos,
        .enabledExtensionCount = sizeof(extensions) / sizeof(extensions[0]),
        .ppEnabledExtensionNames = extensions
    };
    // TODO: check for the extensions instead of bumbling ahead
    if ((vk_result = vkCreateDevice(phys_device, &device_info, NULL, &out->device)) != VK_SUCCESS)
    {
        set_vk_error("vkCreateDevice failed", vk_result);
        goto free_devices;
    }

    VkPhysicalDeviceMemoryProperties memory_properties = {};
    vkGetPhysicalDeviceMemoryProperties(phys_device, &memory_properties);
    uint32_t i;
    VkMemoryPropertyFlags require = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    if (flags & VKGDR_OPEN_REQUIRE_COHERENT_BIT)
        require |= VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (i = 0; i < memory_properties.memoryTypeCount; i++)
    {
        if ((memory_properties.memoryTypes[i].propertyFlags & require) == require)
            break;
    }
    if (i == memory_properties.memoryTypeCount)
    {
        set_generic_error("Vulkan device does not provide a suitable memory type");
        goto free_devices;
    }
    out->memory_type = i;
    if ((memory_properties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        && !(flags & VKGDR_OPEN_FORCE_NON_COHERENT_BIT))
    {
        out->coherent = true;
        out->non_coherent_atom_size = 1;
    }
    else
    {
        VkPhysicalDeviceProperties device_props = {};
        vkGetPhysicalDeviceProperties(phys_device, &device_props);
        out->coherent = false;
        out->non_coherent_atom_size = device_props.limits.nonCoherentAtomSize;
    }

    free(devices);
    out->phys_device = phys_device;

    out->vkDestroyInstance = vkDestroyInstance;
#define INIT_VK_DEVICE_PFN(out, name) (out)->name = (PFN_ ## name) vkGetDeviceProcAddr((out)->device, #name)
    INIT_VK_DEVICE_PFN(out, vkAllocateMemory);
    INIT_VK_DEVICE_PFN(out, vkDestroyDevice);
    INIT_VK_DEVICE_PFN(out, vkFlushMappedMemoryRanges);
    INIT_VK_DEVICE_PFN(out, vkFreeMemory);
    INIT_VK_DEVICE_PFN(out, vkGetMemoryFdKHR);
    INIT_VK_DEVICE_PFN(out, vkInvalidateMappedMemoryRanges);
    INIT_VK_DEVICE_PFN(out, vkMapMemory);
    INIT_VK_DEVICE_PFN(out, vkUnmapMemory);
    return out;

free_devices:
    free(devices);
destroy_instance:
    vkDestroyInstance(out->instance, NULL);
free_out:
    free(out);
    dlclose(libcuda_handle);
close_libvulkan:
    dlclose(libvulkan_handle);
fail:
    return NULL;
#undef INIT_VK_INSTANCE_PFN
#undef INIT_VK_DEVICE_PFN
}

void vkgdr_close(vkgdr_t g)
{
    if (g)
    {
        g->vkDestroyDevice(g->device, NULL);
        g->vkDestroyInstance(g->instance, NULL);
        dlclose(g->libcuda_handle);
        dlclose(g->libvulkan_handle);
        free(g);
    }
}

vkgdr_memory_t vkgdr_memory_alloc(vkgdr_t g, size_t size, uint32_t flags)
{
    VkResult vk_result;
    CUresult cu_result;

    clear_last_error();
    // TODO: only supports Linux
    VkExportMemoryAllocateInfo export_info =
    {
        .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
    };
    VkMemoryAllocateInfo info =
    {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &export_info,
        .allocationSize = size,
        .memoryTypeIndex = g->memory_type
    };
    vkgdr_memory_t out = calloc(1, sizeof(struct vkgdr_memory));
    if ((vk_result = g->vkAllocateMemory(g->device, &info, NULL, &out->memory)) != VK_SUCCESS)
    {
        set_vk_error("vkAllocateMemory failed", vk_result);
        goto free_out;
    }
    out->owner = g;
    out->size = size;

    if ((vk_result = g->vkMapMemory(g->device, out->memory, 0, VK_WHOLE_SIZE, 0, &out->host_ptr)) != VK_SUCCESS)
    {
        set_vk_error("vkMapMemory failed", vk_result);
        goto free_memory;
    }
    int fd = -1;
    VkMemoryGetFdInfoKHR fd_info =
    {
        .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
        .memory = out->memory,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
    };
    if ((vk_result = g->vkGetMemoryFdKHR(g->device, &fd_info, &fd)) != VK_SUCCESS)
    {
        set_vk_error("vkGetMemoryFdKHR failed", vk_result);
        goto unmap_memory;
    }

    const CUDA_EXTERNAL_MEMORY_HANDLE_DESC ext_desc =
    {
        .type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD,
        .handle = {.fd = fd},
        .size = size
    };
    const CUDA_EXTERNAL_MEMORY_BUFFER_DESC buffer_desc =
    {
        .offset = 0,
        .size = size,
        .flags = 0
    };
    if ((cu_result = g->fn_cuImportExternalMemory(&out->ext_mem, &ext_desc)) != CUDA_SUCCESS)
    {
        close(fd);
        set_cu_error("cuImportExternalMemory failed", cu_result);
        goto unmap_memory;
    }
    fd = -1;  // CUDA has taken ownership
    if ((cu_result = g->fn_cuExternalMemoryGetMappedBuffer(&out->device_ptr, out->ext_mem, &buffer_desc)) != CUDA_SUCCESS)
    {
        set_cu_error("cuExternalMemoryGetMappedBuffer failed", cu_result);
        goto destroy_external;
    }

    return out;

destroy_external:
    g->fn_cuDestroyExternalMemory(out->ext_mem);
unmap_memory:
    g->vkUnmapMemory(g->device, out->memory);
free_memory:
    g->vkFreeMemory(g->device, out->memory, NULL);
free_out:
    free(out);
    return NULL;
}

void vkgdr_memory_free(vkgdr_memory_t mem)
{
    if (mem)
    {
        vkgdr_t g = mem->owner;
        g->fn_cuMemFree_v2(mem->device_ptr);
        g->fn_cuDestroyExternalMemory(mem->ext_mem);
        g->vkUnmapMemory(g->device, mem->memory);
        g->vkFreeMemory(g->device, mem->memory, NULL);
        free(mem);
    }
}

void *vkgdr_memory_get_host_ptr(vkgdr_memory_t mem)
{
    return mem->host_ptr;
}

CUdeviceptr vkgdr_memory_get_device_ptr(vkgdr_memory_t mem)
{
    return mem->device_ptr;
}

size_t vkgdr_memory_get_size(vkgdr_memory_t mem)
{
    return mem->size;
}

bool vkgdr_memory_is_coherent(vkgdr_memory_t mem)
{
    return mem->owner->coherent;
}

size_t vkgdr_memory_non_coherent_atom_size(vkgdr_memory_t mem)
{
    return mem->owner->non_coherent_atom_size;
}

void vkgdr_memory_flush(vkgdr_memory_t mem, size_t offset, size_t size)
{
    if (!mem->owner->coherent)
    {
        const VkMappedMemoryRange range =
        {
            .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
            .memory = mem->memory,
            .offset = offset,
            .size = size
        };
        mem->owner->vkFlushMappedMemoryRanges(mem->owner->device, 1, &range);
    }
}

void vkgdr_memory_invalidate(vkgdr_memory_t mem, size_t offset, size_t size)
{
    if (!mem->owner->coherent)
    {
        const VkMappedMemoryRange range =
        {
            .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
            .memory = mem->memory,
            .offset = offset,
            .size = size
        };
        mem->owner->vkInvalidateMappedMemoryRanges(mem->owner->device, 1, &range);
    }
}
