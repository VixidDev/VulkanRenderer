#include "BakedModelLoader.hpp"

#include "../vulkan/VulkanContext.hpp"
#include "../vulkan/objects/VkImage.hpp"
#include "../vulkan/VulkanDevice.hpp"

namespace BakedModelLoader {
	
	std::vector<std::pair<vk::Image, vk::ImageView>> loadTextures(const VulkanContext& context, BakedModel& bakedModel) {
		std::vector<std::pair<vk::Image, vk::ImageView>> textures;

		for (BakedTextureInfo textureInfo : bakedModel.textures) {
			VkFormat format = textureInfo.space == ETextureSpace::srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;

			vk::Image image = vk::loadImage(textureInfo.path.c_str(), context, format, textureInfo.channels);
			vk::ImageView imageView = vk::createImageView(context, image.image, format);

			textures.emplace_back(std::move(image), std::move(imageView));
		}

		return textures;
	}

	std::vector<VkDescriptorSet> createMaterialDescriptors(Driver& driver, BakedModel& bakedModel) {
		std::vector<VkDescriptorSet> materialDescriptors;

		Renderer& renderer = driver.getRenderer();
		VulkanWindow& window = *renderer.getContext().window;

		std::vector<std::pair<vk::Image, vk::ImageView>>& textures = driver.getSceneTextures();

		for (std::size_t i = 0; i < bakedModel.materials.size(); i++) {
			VkDescriptorSet materialDescriptor = VkUtils::createDescriptorSet(
				window,
				window.device->descPool,
				renderer.getDescriptorSetLayout("materials"));

			VkWriteDescriptorSet desc[6]{};

			VkDescriptorImageInfo baseColourInfo{};
			baseColourInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			baseColourInfo.imageView = textures[bakedModel.materials[i].baseColorTextureId].second.handle;
			baseColourInfo.sampler = renderer.getDefaultSampler().handle;

			desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			desc[0].dstSet = materialDescriptor;
			desc[0].dstBinding = 0;
			desc[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			desc[0].descriptorCount = 1;
			desc[0].pImageInfo = &baseColourInfo;

			VkDescriptorImageInfo metalnessInfo{};
			metalnessInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			metalnessInfo.imageView = textures[bakedModel.materials[i].metalnessTextureId].second.handle;
			metalnessInfo.sampler = renderer.getDefaultSampler().handle;

			desc[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			desc[1].dstSet = materialDescriptor;
			desc[1].dstBinding = 1;
			desc[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			desc[1].descriptorCount = 1;
			desc[1].pImageInfo = &metalnessInfo;

			VkDescriptorImageInfo roughnessInfo{};
			roughnessInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			roughnessInfo.imageView = textures[bakedModel.materials[i].roughnessTextureId].second.handle;
			roughnessInfo.sampler = renderer.getDefaultSampler().handle;

			desc[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			desc[2].dstSet = materialDescriptor;
			desc[2].dstBinding = 2;
			desc[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			desc[2].descriptorCount = 1;
			desc[2].pImageInfo = &roughnessInfo;

			// Check if the material has a valid alphaMaskTextureId, otherwise set its
			// imageView handle to the base colour texture
			VkImageView alphaMaskImageView = VK_NULL_HANDLE;
			if (bakedModel.materials[i].alphaMaskTextureId == 0xffffffff)
				alphaMaskImageView = textures[bakedModel.materials[i].baseColorTextureId].second.handle;
			else
				alphaMaskImageView = textures[bakedModel.materials[i].alphaMaskTextureId].second.handle;

			VkDescriptorImageInfo alphaMaskInfo{};
			alphaMaskInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			alphaMaskInfo.imageView = alphaMaskImageView;
			alphaMaskInfo.sampler = renderer.getDefaultSampler().handle;

			desc[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			desc[3].dstSet = materialDescriptor;
			desc[3].dstBinding = 3;
			desc[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			desc[3].descriptorCount = 1;
			desc[3].pImageInfo = &alphaMaskInfo;

			VkDescriptorImageInfo normalMapInfo{};
			normalMapInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			normalMapInfo.imageView = textures[bakedModel.materials[i].normalMapTextureId].second.handle;
			normalMapInfo.sampler = renderer.getDefaultSampler().handle;

			desc[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			desc[4].dstSet = materialDescriptor;
			desc[4].dstBinding = 4;
			desc[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			desc[4].descriptorCount = 1;
			desc[4].pImageInfo = &normalMapInfo;

			VkImageView emissiveImageView = VK_NULL_HANDLE;
			if (bakedModel.materials[i].emissiveTextureId == 0xffffffff)
				emissiveImageView = renderer.getDummyTexture().second.handle;
			else
				emissiveImageView = textures[bakedModel.materials[i].emissiveTextureId].second.handle;

			VkDescriptorImageInfo emissiveInfo{};
			emissiveInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			emissiveInfo.imageView = textures[bakedModel.materials[i].emissiveTextureId].second.handle;
			emissiveInfo.sampler = renderer.getDefaultSampler().handle;

			desc[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			desc[5].dstSet = materialDescriptor;
			desc[5].dstBinding = 5;
			desc[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			desc[5].descriptorCount = 1;
			desc[5].pImageInfo = &emissiveInfo;

			constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
			vkUpdateDescriptorSets(window.device->device, numSets, desc, 0, nullptr);
			materialDescriptors.emplace_back(materialDescriptor);
		}
		
		return materialDescriptors;
	}

	std::vector<MeshData> uploadToGPU(const VulkanContext& context, BakedModel& bakedModel) {
		VulkanWindow& window = *context.window;
		VulkanAllocator& allocator = *context.allocator;
		
		std::vector<MeshData> meshData;

		for (std::size_t i = 0; i < bakedModel.meshes.size(); i++) {
			// GPU sided buffers
			vk::Buffer vertexPosGPU = vk::createBuffer(
				allocator,
				bakedModel.meshes[i].positions.size() * sizeof(glm::vec3),
				VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				0,
				VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

			vk::Buffer vertexTexGPU = vk::createBuffer(
				allocator,
				bakedModel.meshes[i].texcoords.size() * sizeof(glm::vec2),
				VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				0,
				VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

			vk::Buffer vertexNormsGPU = vk::createBuffer(
				allocator,
				bakedModel.meshes[i].normals.size() * sizeof(glm::vec3),
				VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				0,
				VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

			vk::Buffer vertexTBNGPU = vk::createBuffer(
				allocator,
				bakedModel.meshes[i].tangentsComp.size() * sizeof(std::uint32_t),
				VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				0,
				VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

			vk::Buffer vertexIndexGPU = vk::createBuffer(
				allocator,
				bakedModel.meshes[i].indices.size() * sizeof(std::uint32_t),
				VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				0,
				VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

			// Staging buffers
			vk::Buffer posStaging = vk::createBuffer(
				allocator,
				bakedModel.meshes[i].positions.size() * sizeof(glm::vec3),
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
		
			vk::Buffer texStaging = vk::createBuffer(
				allocator,
				bakedModel.meshes[i].texcoords.size() * sizeof(glm::vec2),
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

			vk::Buffer normsStaging = vk::createBuffer(
				allocator,
				bakedModel.meshes[i].normals.size() * sizeof(glm::vec3),
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

			vk::Buffer tbnStaging = vk::createBuffer(
				allocator,
				bakedModel.meshes[i].tangentsComp.size() * sizeof(std::uint32_t),
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

			vk::Buffer indexStaging = vk::createBuffer(
				allocator,
				bakedModel.meshes[i].indices.size() * sizeof(std::uint32_t),
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

			mapToGPU(allocator, vertexPosGPU, posStaging, bakedModel.meshes[i].positions);
			mapToGPU(allocator, vertexTexGPU, texStaging, bakedModel.meshes[i].texcoords);
			mapToGPU(allocator, vertexNormsGPU, normsStaging, bakedModel.meshes[i].normals);
			mapToGPU(allocator, vertexTBNGPU, tbnStaging, bakedModel.meshes[i].tangentsComp);
			mapToGPU(allocator, vertexIndexGPU, indexStaging, bakedModel.meshes[i].indices);

			VkCommandBuffer uploadCmd = VkUtils::createCommandBuffer(window, window.device->cmdPool);

			VkUtils::beginCommandBuffer(uploadCmd);

			copyToGPU(uploadCmd, vertexPosGPU, posStaging, bakedModel.meshes[i].positions);
			copyToGPU(uploadCmd, vertexTexGPU, texStaging, bakedModel.meshes[i].texcoords);
			copyToGPU(uploadCmd, vertexNormsGPU, normsStaging, bakedModel.meshes[i].normals);
			copyToGPU(uploadCmd, vertexTBNGPU, tbnStaging, bakedModel.meshes[i].tangentsComp);
			copyToGPU(uploadCmd, vertexIndexGPU, indexStaging, bakedModel.meshes[i].indices);

			VkUtils::endAndSubmitCommandBuffer(window, uploadCmd);

			bool hasAlphaMask = false;
			if (bakedModel.materials[bakedModel.meshes[i].materialId].alphaMaskTextureId != 0xffffffff) hasAlphaMask = true;

			meshData.emplace_back(
				MeshData{
					std::move(vertexPosGPU),
					std::move(vertexTexGPU),
					std::move(vertexNormsGPU),
					std::move(vertexTBNGPU),
					std::move(vertexIndexGPU),
					bakedModel.meshes[i].indices.size(),
					bakedModel.meshes[i].materialId,
					hasAlphaMask
				});
		}

		return meshData;
	}

}