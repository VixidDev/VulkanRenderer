#include "CameraPlanesUniformBuffer.hpp"

#include "../../../../vulkan/VkUtils.hpp"

CameraPlanesUniformBuffer::CameraPlanesUniformBuffer(
	VulkanAllocator* allocator,
	VkPipelineStageFlags stageFlags,
	glsl::CameraPlanesUniform* cameraPlanesUniform) : UniformBuffer(allocator, stageFlags) {
	this->buffer = vk::createBuffer(
		*this->allocator,
		sizeof(glsl::DepthMVPUniform),
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		0,
		VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

	this->uniformData = cameraPlanesUniform;
}

void CameraPlanesUniformBuffer::update(VkCommandBuffer cmdBuff) {
	Utils::bufferBarrier(
		cmdBuff,
		this->buffer.buffer,
		VK_ACCESS_UNIFORM_READ_BIT,
		VK_ACCESS_TRANSFER_WRITE_BIT,
		this->stageFlags,
		VK_PIPELINE_STAGE_TRANSFER_BIT);

	vkCmdUpdateBuffer(cmdBuff, this->buffer.buffer, 0, sizeof(glsl::CameraPlanesUniform), this->uniformData);

	Utils::bufferBarrier(
		cmdBuff,
		this->buffer.buffer,
		VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_ACCESS_UNIFORM_READ_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		this->stageFlags);
}