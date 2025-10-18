#pragma once

#include "IBuffer.hpp"

class IUniformBuffer : public IBuffer {
public:
	virtual ~IUniformBuffer() = default;
	virtual void update(std::uint32_t frameIndex, VkCommandBuffer cmdBuf) = 0;
};