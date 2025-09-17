#pragma once

class IShaderStorageBuffer {
public:
	virtual ~IShaderStorageBuffer() = default;
	virtual void update(VkCommandBuffer cmdBuff = VK_NULL_HANDLE) = 0;
	virtual std::uint32_t getBufferSize() const = 0;
	virtual VkBuffer getHandle() const = 0;
};