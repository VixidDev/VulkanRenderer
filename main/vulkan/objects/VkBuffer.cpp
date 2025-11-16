#include "VkBuffer.hpp"

#include "Error.hpp"
#include "toString.hpp"

namespace vk {

	Buffer::~Buffer() {
		if (mBuffer != VK_NULL_HANDLE) {
			assert(VK_NULL_HANDLE != mAllocator);
			assert(VK_NULL_HANDLE != mAllocation);
			vmaDestroyBuffer(mAllocator, mBuffer, mAllocation);
		}
	}

	Buffer::Buffer(VmaAllocator allocator, VkBuffer buffer, VmaAllocation allocation) noexcept
		: mBuffer(buffer)
		, mAllocation(allocation)
		, mAllocator(allocator) {}

	Buffer::Buffer(Buffer&& aOther) noexcept
		: mBuffer(std::exchange(aOther.mBuffer, VK_NULL_HANDLE))
		, mAllocation(std::exchange(aOther.mAllocation, VK_NULL_HANDLE))
		, mAllocator(std::exchange(aOther.mAllocator, VK_NULL_HANDLE)) {}

	Buffer& Buffer::operator=(Buffer&& aOther) noexcept {
		std::swap(mBuffer, aOther.mBuffer);
		std::swap(mAllocation, aOther.mAllocation);
		std::swap(mAllocator, aOther.mAllocator);
		return *this;
	}

	Buffer Buffer::createBuffer(
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

		if (const VkResult res = vmaCreateBuffer(allocator.allocator, &bufferInfo, &allocInfo, &buffer, &allocation, nullptr); VK_SUCCESS != res)
			throw Utils::Error("Unable to allocate buffer\n vmaCreateBuffer() returned %s", Utils::toString(res).c_str());

		return Buffer(allocator.allocator, buffer, allocation);
	}

}