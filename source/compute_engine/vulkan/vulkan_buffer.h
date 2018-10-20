#pragma once

#include <cstring>
#include "vulkan/vulkan.h"
#include "vulkan_utility.h"

namespace ktt
{

class VulkanBuffer
{
public:
    VulkanBuffer() :
        device(nullptr),
        buffer(nullptr),
        bufferMemory(nullptr),
        bufferSize(0),
        usageFlags(VK_BUFFER_USAGE_TRANSFER_SRC_BIT)
    {}

    explicit VulkanBuffer(VkDevice device, const VkDeviceSize bufferSize, const VkBufferUsageFlags usageFlags) :
        device(device),
        bufferMemory(nullptr),
        bufferSize(bufferSize),
        usageFlags(usageFlags)
    {
        const VkBufferCreateInfo bufferCreateInfo =
        {
            VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            nullptr,
            0,
            bufferSize,
            usageFlags,
            VK_SHARING_MODE_EXCLUSIVE,
            0,
            nullptr
        };

        checkVulkanError(vkCreateBuffer(device, &bufferCreateInfo, nullptr, &buffer), "vkCreateBuffer");
    }

    ~VulkanBuffer()
    {
        if (buffer != nullptr)
        {
            vkDestroyBuffer(device, buffer, nullptr);
        }
        
        if (bufferMemory != nullptr)
        {
            vkFreeMemory(device, bufferMemory, nullptr);
        }
    }

    VkMemoryRequirements getMemoryRequirements() const
    {
        VkMemoryRequirements requirements;
        vkGetBufferMemoryRequirements(device, buffer, &requirements);
        return requirements;
    }

    void allocateMemory(const uint32_t memoryTypeIndex)
    {
        VkMemoryRequirements memoryRequirements = getMemoryRequirements();

        const VkMemoryAllocateInfo memoryAllocateInfo =
        {
            VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            nullptr,
            memoryRequirements.size,
            memoryTypeIndex
        };

        checkVulkanError(vkAllocateMemory(device, &memoryAllocateInfo, nullptr, &bufferMemory), "vkAllocateMemory");
        checkVulkanError(vkBindBufferMemory(device, buffer, bufferMemory, 0), "vkBindBufferMemory");
    }

    void uploadData(const void* source, const VkDeviceSize dataSize)
    {
        void* data;
        checkVulkanError(vkMapMemory(device, bufferMemory, 0, dataSize, 0, &data), "vkMapMemory");
        std::memcpy(data, source, dataSize);
        vkUnmapMemory(device, bufferMemory);
    }

    void uploadData(VkBuffer sourceBuffer, const VkDeviceSize dataSize, VkCommandBuffer commandBuffer)
    {
        const VkCommandBufferBeginInfo commandBufferBeginInfo =
        {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            nullptr,
            VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            nullptr
        };

        const VkBufferCopy copyRegion =
        {
            0,
            0,
            dataSize
        };

        checkVulkanError(vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo), "vkBeginCommandBuffer");
        vkCmdCopyBuffer(commandBuffer, sourceBuffer, buffer, 1, &copyRegion);
        checkVulkanError(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer");
    }

    VkDevice getDevice() const
    {
        return device;
    }

    VkBuffer getBuffer() const
    {
        return buffer;
    }

    VkDeviceSize getBufferSize() const
    {
        return bufferSize;
    }

    VkBufferUsageFlags getUsageFlags() const
    {
        return usageFlags;
    }

private:
    VkDevice device;
    VkBuffer buffer;
    VkDeviceMemory bufferMemory;
    VkDeviceSize bufferSize;
    VkBufferUsageFlags usageFlags;
};

} // namespace ktt
