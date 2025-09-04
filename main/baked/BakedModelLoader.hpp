#pragma once

#include <vector>

#include "BakedModel.hpp"
#include "../Driver.hpp"
#include "../vulkan/objects/VkObjects.hpp"

namespace BakedModelLoader {

	std::vector<std::pair<vk::Image, vk::ImageView>> loadTextures(const VulkanContext& context, BakedModel& bakedModel);

	std::vector<VkDescriptorSet> createMaterialDescriptors(Driver& driver, BakedModel& bakedModel);

	std::vector<MeshData> uploadToGPU(const VulkanContext& context, BakedModel& bakedModel);

	template <class T>
	void mapToGPU(const VulkanAllocator& allocator, vk::Buffer& gpuBuffer, vk::Buffer& stagingBuffer, std::vector<T>& vertexAttribute);

	template <class T>
	void copyToGPU(VkCommandBuffer cmdBuff, vk::Buffer& gpuBuffer, vk::Buffer& stagingBuffer, std::vector<T>& vertexAttribute);
}