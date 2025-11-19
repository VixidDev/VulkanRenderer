#pragma once

#include "../VulkanAllocator.hpp"

#include <cassert>

namespace vk {

	class Buffer {
	public:
		Buffer() noexcept = default;
		~Buffer();

		Buffer(Buffer const&) = delete;
		Buffer& operator=(Buffer const&) = delete;

		Buffer(Buffer&&) noexcept;
		Buffer& operator=(Buffer&&) noexcept;

		static Buffer createBuffer(
			const VulkanAllocator& allocator,
			VkDeviceSize size,
			VkBufferUsageFlags bufferUsage,
			VmaAllocationCreateFlags memoryFlags,
			VmaMemoryUsage usageFlags = VMA_MEMORY_USAGE_AUTO
		);

		explicit operator bool() const { return mBuffer != VK_NULL_HANDLE; }

		VkBuffer get() const noexcept { return mBuffer; }
		VmaAllocation getAllocation() const noexcept { return mAllocation; }

	private:
		explicit Buffer(VmaAllocator allocator, VkBuffer buffer = VK_NULL_HANDLE, VmaAllocation allocation = VK_NULL_HANDLE) noexcept;

		VkBuffer mBuffer = VK_NULL_HANDLE;
		VmaAllocation mAllocation = VK_NULL_HANDLE;
		VmaAllocator mAllocator = VK_NULL_HANDLE;
	};

}