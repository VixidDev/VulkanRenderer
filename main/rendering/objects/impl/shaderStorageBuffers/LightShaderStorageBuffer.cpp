#include "LightShaderStorageBuffer.hpp"

#include "../../../../vulkan/VkUtils.hpp"
#include "../../../../vulkan/VulkanDevice.hpp"

#include "Error.hpp"
#include "toString.hpp"

// This is near identical to LightMatricesShaderStorageBuffer, I could probably just use a general shader storage
// buffer class rather than individual implementations and just template the type for the buffer size
LightShaderStorageBuffer::LightShaderStorageBuffer(
	VulkanContext* context,
	std::vector<glsl::Light>* lights) : ssboData(lights), ShaderStorageBuffer(context)
{
	this->bufferSize = this->ssboData->size() * sizeof(glsl::Light);
	
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

void LightShaderStorageBuffer::update(VkCommandBuffer cmdBuff) {
	// Map ptr to GPU and copy to it
	void* ptr;
	if (const auto res = vmaMapMemory(this->context->allocator->allocator, stagingBuffer.allocation, &ptr); VK_SUCCESS != res)
		throw Utils::Error("Mapping memory for writing to Lights SSBO\nvmaMapMemory() returned: %s\n", Utils::toString(res).c_str());

	std::memcpy(ptr, this->ssboData->data(), this->bufferSize);
	vmaUnmapMemory(this->context->allocator->allocator, stagingBuffer.allocation);

	auto copyCommand = [this](VkCommandBuffer cmdBuff) {
		VkBufferCopy copyRegion = {
			.size = this->bufferSize
		};

		vkCmdCopyBuffer(cmdBuff, this->stagingBuffer.buffer, this->gpuBuffer.buffer, 1, &copyRegion);
	};

	// Upload to GPU
	VkCommandBuffer uploadCmdBuff = cmdBuff;
	if (uploadCmdBuff == VK_NULL_HANDLE) {
		uploadCmdBuff = createCommandBuffer(*this->context->window);
		beginCommandBuffer(uploadCmdBuff);
		
		copyCommand(uploadCmdBuff);

		endAndSubmitCommandBuffer(*this->context->window, uploadCmdBuff);
	} else {
		copyCommand(cmdBuff);
	}
}