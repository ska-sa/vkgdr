#define VK_NO_PROTOTYPES 1  /* Prevents Vulkan headers from defining functions to link to */

#include <unistd.h>
#include <assert.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <dlfcn.h>
#include <cuda.h>
#include <vulkan/vulkan.h>
#include <dlfcn.h>
#include "vkgdr.h"

struct vkgdr
{
    void *libvulkan_handle;
    VkInstance instance;
    VkPhysicalDevice phys_device;
    VkDevice device;
    uint32_t memory_type;
    bool coherent;
    size_t non_coherent_atom_size;

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

vkgdr_t vkgdr_open(CUdevice device, uint32_t flags)
{
    void *libvulkan_handle = dlopen("libvulkan.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (!libvulkan_handle)
        goto fail;
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr) dlsym(libvulkan_handle, "vkGetInstanceProcAddr");
#define INIT_INSTANCE_PFN(instance, name) PFN_ ## name name = (PFN_ ## name) vkGetInstanceProcAddr((instance), #name)
    INIT_INSTANCE_PFN(NULL, vkCreateInstance);

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
    if (vkCreateInstance(&instance_info, NULL, &out->instance) != VK_SUCCESS)
        goto free_out;

    INIT_INSTANCE_PFN(out->instance, vkEnumeratePhysicalDevices);
    INIT_INSTANCE_PFN(out->instance, vkGetPhysicalDeviceProperties2);
    INIT_INSTANCE_PFN(out->instance, vkCreateDevice);
    INIT_INSTANCE_PFN(out->instance, vkGetPhysicalDeviceProperties);
    INIT_INSTANCE_PFN(out->instance, vkGetPhysicalDeviceMemoryProperties);
    INIT_INSTANCE_PFN(out->instance, vkDestroyInstance);
    INIT_INSTANCE_PFN(out->instance, vkGetDeviceProcAddr);

    // Find the matching Vulkan physical device
    if (cuDeviceGetUuid(&uuid, device) != CUDA_SUCCESS)
        goto destroy_instance;

    uint32_t n_devices;
    if (vkEnumeratePhysicalDevices(out->instance, &n_devices, NULL) != VK_SUCCESS)
        goto destroy_instance;
    VkPhysicalDevice *devices = calloc(n_devices, sizeof(VkPhysicalDevice));
    if (vkEnumeratePhysicalDevices(out->instance, &n_devices, devices) != VK_SUCCESS)
        goto free_devices;
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
        goto free_devices;

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
    if (vkCreateDevice(phys_device, &device_info, NULL, &out->device) != VK_SUCCESS)
        goto free_devices;

    VkPhysicalDeviceMemoryProperties memory_properties = {};
    vkGetPhysicalDeviceMemoryProperties(phys_device, &memory_properties);
    uint32_t i;
    for (i = 0; i < memory_properties.memoryTypeCount; i++)
    {
        const VkMemoryType *t = &memory_properties.memoryTypes[i];
        const VkMemoryPropertyFlags require = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        if ((t->propertyFlags & require) == require)
            break;
    }
    if (i == memory_properties.memoryTypeCount)
        goto free_devices;  // no suitable memory type found
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
#define INIT_DEVICE_PFN(out, name) (out)->name = (PFN_ ## name) vkGetDeviceProcAddr((out)->device, #name)
    INIT_DEVICE_PFN(out, vkAllocateMemory);
    INIT_DEVICE_PFN(out, vkDestroyDevice);
    INIT_DEVICE_PFN(out, vkFlushMappedMemoryRanges);
    INIT_DEVICE_PFN(out, vkFreeMemory);
    INIT_DEVICE_PFN(out, vkGetMemoryFdKHR);
    INIT_DEVICE_PFN(out, vkInvalidateMappedMemoryRanges);
    INIT_DEVICE_PFN(out, vkMapMemory);
    INIT_DEVICE_PFN(out, vkUnmapMemory);
    return out;

free_devices:
    free(devices);
destroy_instance:
    vkDestroyInstance(out->instance, NULL);
free_out:
    free(out);
    dlclose(libvulkan_handle);
fail:
    return NULL;
#undef INIT_INSTANCE_PFN
}

vkgdr_t vkgdr_open_current(uint32_t flags)
{
    CUdevice device;
    if (cuCtxGetDevice(&device) != CUDA_SUCCESS)
        return NULL;
    return vkgdr_open(device, flags);
}

void vkgdr_close(vkgdr_t g)
{
    if (g)
    {
        g->vkDestroyDevice(g->device, NULL);
        g->vkDestroyInstance(g->instance, NULL);
        dlclose(g->libvulkan_handle);
        free(g);
    }
}

vkgdr_memory_t vkgdr_memory_alloc(vkgdr_t g, size_t size, uint32_t flags)
{
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
    if (g->vkAllocateMemory(g->device, &info, NULL, &out->memory) != VK_SUCCESS)
        goto free_out;
    out->owner = g;
    out->size = size;

    if (g->vkMapMemory(g->device, out->memory, 0, VK_WHOLE_SIZE, 0, &out->host_ptr) != VK_SUCCESS)
        goto free_memory;

    int fd = -1;
    VkMemoryGetFdInfoKHR fd_info =
    {
        .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
        .memory = out->memory,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
    };
    if (g->vkGetMemoryFdKHR(g->device, &fd_info, &fd) != VK_SUCCESS)
        goto unmap_memory;

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
    if (cuImportExternalMemory(&out->ext_mem, &ext_desc) != CUDA_SUCCESS)
    {
        close(fd);
        goto unmap_memory;
    }
    fd = -1;  // CUDA has taken ownership
    if (cuExternalMemoryGetMappedBuffer(&out->device_ptr, out->ext_mem, &buffer_desc) != CUDA_SUCCESS)
        goto destroy_external;

    return out;

destroy_external:
    cuDestroyExternalMemory(out->ext_mem);
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
        cuMemFree(mem->device_ptr);
        cuDestroyExternalMemory(mem->ext_mem);
        mem->owner->vkUnmapMemory(mem->owner->device, mem->memory);
        mem->owner->vkFreeMemory(mem->owner->device, mem->memory, NULL);
        free(mem);
    }
}

void *vkgdr_memory_get_host(vkgdr_memory_t mem)
{
    return mem->host_ptr;
}

CUdeviceptr vkgdr_memory_get_device(vkgdr_memory_t mem)
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
