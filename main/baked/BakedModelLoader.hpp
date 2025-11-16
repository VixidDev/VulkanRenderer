#pragma once

#include "Error.hpp"
#include "toString.hpp"
#include "../Driver.hpp"
#include "../vulkan/objects/VkObjects.hpp"
#include "../vulkan/VkUtils.hpp"
#include "../models/MeshData.hpp"

namespace BakedModelLoader {

	std::vector<std::pair<vk::Image, vk::ImageView>> loadTextures(const VulkanContext& context, BakedModel& bakedModel);

	std::vector<VkDescriptorSet> createMaterialDescriptors(Driver& driver, BakedModel& bakedModel);

	std::vector<MeshData> uploadToGPU(const VulkanContext& context, BakedModel& bakedModel);

	template <class T>
	void mapToGPU(const VulkanAllocator& allocator, vk::Buffer& gpuBuffer, vk::Buffer& stagingBuffer, std::vector<T>& vertexAttribute) {
		void* ptr = nullptr;

		if (const auto res = vmaMapMemory(allocator.allocator, stagingBuffer.getAllocation(), &ptr); VK_SUCCESS != res)
			throw Utils::Error("Mapping memory for writing\n vmaMapMemory() returned %s\n", Utils::toString(res).c_str());

		std::memcpy(ptr, vertexAttribute.data(), vertexAttribute.size() * sizeof(T));
		vmaUnmapMemory(allocator.allocator, stagingBuffer.getAllocation());
	}

	template<class T>
	void copyToGPU(VkCommandBuffer cmdBuff, vk::Buffer& gpuBuffer, vk::Buffer& stagingBuffer, std::vector<T>& vertexAttribute) {
		VkBufferCopy copy{};
		copy.size = vertexAttribute.size() * sizeof(T);

		vkCmdCopyBuffer(cmdBuff, stagingBuffer.get(), gpuBuffer.get(), 1, &copy);

		VkUtils::bufferBarrier(
			cmdBuff,
			gpuBuffer.get(),
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);
	}
}