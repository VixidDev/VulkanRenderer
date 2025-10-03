#pragma once

#include "interfaces/IShaderStorageBuffer.hpp"
#include "../../../vulkan/objects/VkObjects.hpp"
#include "../../../vulkan/objects/VkBuffer.hpp"
#include "../../../vulkan/VkUtils.hpp"
#include "../../../vulkan/VulkanContext.hpp"
#include "../../../vulkan/VulkanDevice.hpp"

#include "Error.hpp"
#include "toString.hpp"

template <class T>
class ShaderStorageBuffer : public IShaderStorageBuffer {
public:
	ShaderStorageBuffer() = default;
	~ShaderStorageBuffer() = default;

	ShaderStorageBuffer(VulkanContext* context, std::vector<T>* data) : context(context), ssboData(data) {
		if (this->ssboData->size() <= 0) {
			this->bufferSize = 0;
			return;
		}

		this->bufferSize = this->ssboData->size() * sizeof(T);

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

	void update(VkCommandBuffer cmdBuff = VK_NULL_HANDLE) override {
		if (this->getHandle() == VK_NULL_HANDLE) return;

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
			uploadCmdBuff = VkUtils::createCommandBuffer(*this->context->window, this->context->window->getDevice()->getCmdPool());
			VkUtils::beginCommandBuffer(uploadCmdBuff);

			copyCommand(uploadCmdBuff);

			VkUtils::endAndSubmitCommandBuffer(*this->context->window, uploadCmdBuff);
		} else {
			copyCommand(cmdBuff);
		}
	}

	std::uint32_t getBufferSize() const override {
		return static_cast<std::uint32_t>(this->bufferSize);
	}

	VkBuffer getHandle() const override {
		return this->gpuBuffer.buffer;
	}

private:
	VulkanContext* context;
	std::vector<T>* ssboData;

	std::size_t bufferSize = 0;

	vk::Buffer gpuBuffer;
	vk::Buffer stagingBuffer;
};