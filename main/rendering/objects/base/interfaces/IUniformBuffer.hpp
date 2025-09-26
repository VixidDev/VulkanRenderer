#pragma once

class IUniformBuffer {
public:
	virtual ~IUniformBuffer() = default;
	virtual void update(VkCommandBuffer cmdBuf) = 0;
	virtual VkBuffer getHandle() const = 0;
};