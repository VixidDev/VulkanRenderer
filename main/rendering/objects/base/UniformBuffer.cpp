#include "UniformBuffer.hpp"

UniformBuffer::UniformBuffer(VulkanAllocator* allocator, VkPipelineStageFlags stageFlags) 
	: allocator(allocator), stageFlags(stageFlags) {}

void UniformBuffer::update(VkCommandBuffer cmdBuff) {}

VkBuffer UniformBuffer::getHandle() {
	return this->buffer.buffer;
}