#include <unistd.h>
#include <assert.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <cuda.h>
#include <vulkan/vulkan.h>
#include "vkgdr.h"

struct vkgdr
{
    VkInstance instance;
    VkPhysicalDevice phys_device;
    VkDevice device;
    uint32_t memory_type;
    PFN_vkGetMemoryFdKHR vkGetMemoryFdKHR;
};

struct vkgdr_memory
{
    vkgdr_t owner;
    VkDeviceMemory memory;
    void *host_ptr;
    CUexternalMemory ext_mem;
    CUdeviceptr device_ptr;
};

vkgdr_t vkgdr_open(CUdevice device)
{
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
    if (vkCreateInstance(&instance_info, NULL, &out->instance) != VK_SUCCESS)
        goto free_out;

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

    free(devices);
    out->phys_device = phys_device;
    out->vkGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR) vkGetDeviceProcAddr(out->device, "vkGetMemoryFdKHR");
    assert(out->vkGetMemoryFdKHR);
    return out;

free_devices:
    free(devices);
destroy_instance:
    vkDestroyInstance(out->instance, NULL);
free_out:
    free(out);
    return NULL;
}

vkgdr_t vkgdr_open_current(void)
{
    CUdevice device;
    if (cuCtxGetDevice(&device) != CUDA_SUCCESS)
        return NULL;
    return vkgdr_open(device);
}

void vkgdr_close(vkgdr_t g)
{
    if (g)
    {
        vkDestroyDevice(g->device, NULL);
        vkDestroyInstance(g->instance, NULL);
        free(g);
    }
}

vkgdr_memory_t vkgdr_malloc(vkgdr_t g, size_t size)
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
    if (vkAllocateMemory(g->device, &info, NULL, &out->memory) != VK_SUCCESS)
        goto free_out;
    out->owner = g;

    if (vkMapMemory(g->device, out->memory, 0, VK_WHOLE_SIZE, 0, &out->host_ptr) != VK_SUCCESS)
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
    vkUnmapMemory(g->device, out->memory);
free_memory:
    vkFreeMemory(g->device, out->memory, NULL);
free_out:
    free(out);
    return NULL;
}

void vkgdr_free(vkgdr_memory_t mem)
{
    if (mem)
    {
        cuMemFree(mem->device_ptr);
        cuDestroyExternalMemory(mem->ext_mem);
        vkUnmapMemory(mem->owner->device, mem->memory);
        vkFreeMemory(mem->owner->device, mem->memory, NULL);
        free(mem);
    }
}

void *vkgdr_get_host(vkgdr_memory_t mem)
{
    return mem->host_ptr;
}

CUdeviceptr vkgdr_get_device(vkgdr_memory_t mem)
{
    return mem->device_ptr;
}
