#include "MVPUniformBuffer.hpp"

#include "../../../../vulkan/VkUtils.hpp"

MVPUniformBuffer::MVPUniformBuffer(
	VulkanAllocator* allocator, 
	VkPipelineStageFlags stageFlags, 
	glsl::MVPUniform* mvpUniform) : UniformBuffer(allocator, stageFlags) 
{
	this->buffer = vk::createBuffer(
		*this->allocator,
		sizeof(glsl::MVPUniform),
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		0,
		VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

	this->uniformData = mvpUniform;
}

void MVPUniformBuffer::update(VkCommandBuffer cmdBuff) {
	Utils::bufferBarrier(
		cmdBuff,
		this->buffer.buffer,
		VK_ACCESS_UNIFORM_READ_BIT,
		VK_ACCESS_TRANSFER_WRITE_BIT,
		this->stageFlags,
		VK_PIPELINE_STAGE_TRANSFER_BIT);

	vkCmdUpdateBuffer(cmdBuff, this->buffer.buffer, 0, sizeof(glsl::MVPUniform), this->uniformData);

	Utils::bufferBarrier(
		cmdBuff,
		this->buffer.buffer,
		VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_ACCESS_UNIFORM_READ_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		this->stageFlags);
}