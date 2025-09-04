#pragma once

#include <volk/volk.h>
#include <vk_mem_alloc.h>

#include <utility>

#include <cassert>

#include "../VulkanAllocator.hpp"

namespace vk {

	class Buffer {
	public:
		Buffer() noexcept, ~Buffer();

		explicit Buffer(VmaAllocator allocator, VkBuffer buffer = VK_NULL_HANDLE, VmaAllocation allocation = VK_NULL_HANDLE) noexcept;

		Buffer(Buffer const&) = delete;
		Buffer& operator= (Buffer const&) = delete;

		Buffer(Buffer&&) noexcept;
		Buffer& operator = (Buffer&&) noexcept;

	public:
		VkBuffer buffer = VK_NULL_HANDLE;
		VmaAllocation allocation = VK_NULL_HANDLE;

	private:
		VmaAllocator mAllocator = VK_NULL_HANDLE;
	};

	Buffer createBuffer(
		const VulkanAllocator& allocator,
		VkDeviceSize size,
		VkBufferUsageFlags bufferUsage,
		VmaAllocationCreateFlags memoryFlags,
		VmaMemoryUsage usageFlags = VMA_MEMORY_USAGE_AUTO
	);

}