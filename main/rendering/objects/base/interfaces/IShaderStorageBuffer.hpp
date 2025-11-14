#pragma once

#include "IBuffer.hpp"

#include <vector>

namespace vk {
	class Buffer;
}

class IShaderStorageBuffer : public IBuffer {
public:
	virtual ~IShaderStorageBuffer() = default;
	virtual void update(std::uint32_t frameIndex, VkCommandBuffer cmdBuff = VK_NULL_HANDLE) = 0;
	virtual std::uint32_t getBufferSize() const = 0;
	virtual std::vector<vk::Buffer>& getBuffers() = 0;
};