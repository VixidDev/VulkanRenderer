#pragma once

#include "interfaces/IShaderStorageBuffer.hpp"
#include "../../../vulkan/objects/VkObjects.hpp"
#include "../../../vulkan/objects/VkBuffer.hpp"
#include "../../../vulkan/VkUtils.hpp"
#include "../../../vulkan/VulkanContext.hpp"
#include "../../../vulkan/VulkanDevice.hpp"

#include "Error.hpp"
#include "toString.hpp"

// Note to self: When getting around to only performing an update when the
// underlying data actually changes (through some dirty flag etc.) make sure
// each gpu buffer has a dirty flag also, since in the case of 3 frames in
// flight, it will take 3 frames to have all buffers contain the up to date
// data. This dirty flag on the gpu buffers will need to be set true every
// time the underlying data is changed, even if a buffer hasn't been un-dirtied
// from a previous data change as the underlying data could change in 2 consecutive
// frames before the 3rd buffer has had a chance to update it self.
// Same goes for UniformBuffers.

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

		// We need a SSBO buffer for each frame in flight as a new frame
		// might start and need to write / read from / to an SSBO when
		// the last frame hasn't finished with it yet
		for (int i = 0; i < Swapchain::MAX_FRAMES_IN_FLIGHT; i++) {
			// GPU-sided buffer
			this->gpuBuffers.emplace_back(vk::createBuffer(
				*this->context->allocator,
				this->bufferSize,
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				0,
				VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE));

			// Staging buffer
			this->stagingBuffers.emplace_back(vk::createBuffer(
				*this->context->allocator,
				this->bufferSize,
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT));
		}
	}

	void update(std::uint32_t frameIndex, VkCommandBuffer cmdBuff = VK_NULL_HANDLE) override {
		if (this->getHandle(frameIndex) == VK_NULL_HANDLE) return;

		assert(frameIndex < this->gpuBuffers.size());
		assert(frameIndex < this->stagingBuffers.size());

		// Map ptr to GPU and copy to it
		void* ptr;
		if (const auto res = vmaMapMemory(this->context->allocator->allocator, this->stagingBuffers[frameIndex].allocation, &ptr); VK_SUCCESS != res)
			throw Utils::Error("Mapping memory for writing to Lights SSBO\nvmaMapMemory() returned: %s\n", Utils::toString(res).c_str());

		std::memcpy(ptr, this->ssboData->data(), this->bufferSize);
		vmaUnmapMemory(this->context->allocator->allocator, this->stagingBuffers[frameIndex].allocation);

		auto copyCommand = [this](std::uint32_t frameIndex, VkCommandBuffer cmdBuff) {
			VkBufferCopy copyRegion = {
				.size = this->bufferSize
			};

			vkCmdCopyBuffer(cmdBuff, this->stagingBuffers[frameIndex].buffer, this->gpuBuffers[frameIndex].buffer, 1, &copyRegion);

			VkUtils::bufferBarrier(
				cmdBuff,
				this->gpuBuffers[frameIndex].buffer,
				/* srcAccessMask */ VK_ACCESS_TRANSFER_WRITE_BIT,
				/* dstAccessMask */ VK_ACCESS_SHADER_READ_BIT,
				/* srcStageMask */ VK_PIPELINE_STAGE_TRANSFER_BIT,
				/* dstStageMask */ VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT); // Pass in stage flags in constructor in case SSBO is in a different stage
		};

		// Upload to GPU
		VkCommandBuffer uploadCmdBuff = cmdBuff;
		if (uploadCmdBuff == VK_NULL_HANDLE) {
			uploadCmdBuff = VkUtils::createCommandBuffer(*this->context->window, this->context->window->getDevice()->getCmdPool());
			VkUtils::beginCommandBuffer(uploadCmdBuff);

			copyCommand(frameIndex, uploadCmdBuff);

			VkUtils::endAndSubmitCommandBuffer(*this->context->window, uploadCmdBuff);
		} else {
			copyCommand(frameIndex, cmdBuff);
		}
	}

	std::uint32_t getBufferSize() const override {
		return static_cast<std::uint32_t>(this->bufferSize);
	}

	VkBuffer getHandle(std::uint32_t frameIndex) const override {
		return this->gpuBuffers.at(frameIndex).buffer;
	}

	std::vector<vk::Buffer>& getBuffers() {
		return this->gpuBuffers;
	}

private:
	VulkanContext* context;
	std::vector<T>* ssboData;

	std::size_t bufferSize = 0;

	std::vector<vk::Buffer> gpuBuffers;
	std::vector<vk::Buffer> stagingBuffers;
};