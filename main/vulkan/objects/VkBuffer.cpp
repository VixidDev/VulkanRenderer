#include "VkBuffer.hpp"

#include <utility>

#include <cassert>

#include "Error.hpp"
#include "toString.hpp"

namespace vk {

	Buffer::Buffer() noexcept = default;

	Buffer::~Buffer() {
		if (VK_NULL_HANDLE != buffer) {
			assert(VK_NULL_HANDLE != mAllocator);
			assert(VK_NULL_HANDLE != allocation);
			vmaDestroyBuffer(mAllocator, buffer, allocation);
		}
	}

	Buffer::Buffer(VmaAllocator allocator, VkBuffer buffer, VmaAllocation allocation) noexcept
		: buffer(buffer)
		, allocation(allocation)
		, mAllocator(allocator) {}

	Buffer::Buffer(Buffer&& aOther) noexcept
		: buffer(std::exchange(aOther.buffer, VK_NULL_HANDLE))
		, allocation(std::exchange(aOther.allocation, VK_NULL_HANDLE))
		, mAllocator(std::exchange(aOther.mAllocator, VK_NULL_HANDLE)) {}
	Buffer& Buffer::operator=(Buffer&& aOther) noexcept {
		std::swap(buffer, aOther.buffer);
		std::swap(allocation, aOther.allocation);
		std::swap(mAllocator, aOther.mAllocator);
		return *this;
	}

	Buffer createBuffer(
		const VulkanAllocator& allocator,
		VkDeviceSize size,
		VkBufferUsageFlags bufferUsage,
		VmaAllocationCreateFlags memoryFlags,
		VmaMemoryUsage memoryUsage) {

		VkBufferCreateInfo bufferInfo{};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = size;
		bufferInfo.usage = bufferUsage;

		VmaAllocationCreateInfo allocInfo{};
		allocInfo.flags = memoryFlags;
		allocInfo.usage = memoryUsage;

		VkBuffer buffer = VK_NULL_HANDLE;
		VmaAllocation allocation = VK_NULL_HANDLE;

		if (const auto res = vmaCreateBuffer(allocator.allocator, &bufferInfo, &allocInfo, &buffer, &allocation, nullptr); VK_SUCCESS != res)
			throw Utils::Error("Unable to allocate buffer\n vmaCreateBuffer() returned %s", Utils::toString(res).c_str());

		return Buffer(allocator.allocator, buffer, allocation);
	}

}