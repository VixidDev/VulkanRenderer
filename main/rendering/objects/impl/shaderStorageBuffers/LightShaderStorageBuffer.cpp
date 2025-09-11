#include "LightShaderStorageBuffer.hpp"

#include "../../../../vulkan/VkUtils.hpp"
#include "../../../../vulkan/VulkanDevice.hpp"

#include "Error.hpp"
#include "toString.hpp"

LightShaderStorageBuffer::LightShaderStorageBuffer(
	VulkanContext* context,
	std::vector<glsl::Light>* lightsUniform) : ssboData(lightsUniform), ShaderStorageBuffer(context)
{
	this->bufferSize = this->ssboData->size() * sizeof(glsl::Light);
	std::fprintf(stdout, "Buffer size for Lights is: %zu - ssboData.size(): %zu - sizeof(glsl::Light): %zu\n", this->bufferSize, this->ssboData->size(), sizeof(glsl::Light));

	// GPU-sided buffer
	this->gpuBuffer = vk::createBuffer(
		*this->context->allocator,
		this->bufferSize,
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		0,
		VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
	);

	// Staging buffer
	this->stagingBuffer = vk::createBuffer(
		*this->context->allocator,
		this->bufferSize,
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
	);

	this->update();
}

void LightShaderStorageBuffer::update() {
	// Map ptr to GPU and copy to it
	void* ptr;
	if (const auto res = vmaMapMemory(this->context->allocator->allocator, stagingBuffer.allocation, &ptr); VK_SUCCESS != res)
		throw Utils::Error("Mapping memory for writing to Lights SSBO\nvmaMapMemory() returned: %s\n", Utils::toString(res).c_str());

	std::memcpy(ptr, this->ssboData->data(), this->bufferSize);
	vmaUnmapMemory(this->context->allocator->allocator, stagingBuffer.allocation);

	// Upload to GPU
	VkCommandBuffer uploadCmdBuff = createCommandBuffer(*this->context->window);

	beginCommandBuffer(uploadCmdBuff);

	VkBufferCopy copyRegion = {
		.size = this->bufferSize
	};

	vkCmdCopyBuffer(uploadCmdBuff, this->stagingBuffer.buffer, this->gpuBuffer.buffer, 1, &copyRegion);

	endAndSubmitCommandBuffer(*this->context->window, uploadCmdBuff);
}