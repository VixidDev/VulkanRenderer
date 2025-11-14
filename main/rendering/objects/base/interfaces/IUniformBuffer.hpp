#pragma once

#include "IBuffer.hpp"

#include <vector>

namespace vk {
	class Buffer;
}

class IUniformBuffer : public IBuffer {
public:
	virtual ~IUniformBuffer() = default;
	virtual void update(std::uint32_t frameIndex, VkCommandBuffer cmdBuf) = 0;
	virtual std::vector<vk::Buffer>& getBuffers() = 0;
};