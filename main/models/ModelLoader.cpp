#define TINYOBJLOADER_IMPLEMENTATION
#include "ModelLoader.hpp"

#include "tiny_obj_loader.h"
#include "../vulkan/VulkanDevice.hpp"

namespace ModelLoader {

	int loadObjFromFile(const VulkanContext& context, const std::string& objFilename, MeshData& meshData) {
		tinyobj::ObjReader reader;

		if (!reader.ParseFromFile(objFilename)) {
			if (!reader.Error().empty()) {
				std::fprintf(stderr, "ModelLoader::loadFromFile(): File '%s' encountered error: '%s'\n", objFilename.c_str(), reader.Error().c_str());
			}
			return 0;
		}

		if (!reader.Warning().empty()) {
			std::fprintf(stdout, "ModelLoader::loadFromFile(): File '%s' encountered warning: '%s'\n", objFilename.c_str(), reader.Warning().c_str());
		}

		const tinyobj::attrib_t& attrib = reader.GetAttrib();
		const std::vector<tinyobj::shape_t>& shapes = reader.GetShapes();

		RawMeshData rawData{};

		// Try get a name
		rawData.name = shapes[0].name;

		// Copy vertices
		for (size_t i = 0; i < attrib.vertices.size(); i += 3) {
			glm::vec3 position;
			position[0] = attrib.vertices[i];
			position[1] = attrib.vertices[i+1];
			position[2] = attrib.vertices[i+2];
			rawData.positions.emplace_back(position);
		}

		// Assume 1 shape
		// Copy indices
		for (size_t i = 0; i < shapes[0].mesh.indices.size(); i++) {
			rawData.indices.emplace_back(shapes[0].mesh.indices[i].vertex_index);
		}

		// Upload to GPU
		meshData = uploadToGPU(context, rawData);

		return 1;
	}
	 
	MeshData uploadToGPU(const VulkanContext& context, RawMeshData& rawData) {
		VulkanWindow& window = *context.window;
		VulkanAllocator& allocator = *context.allocator;

		MeshData meshData{};

		// Positions
		vk::Buffer posBufferGPU{};
		vk::Buffer posBufferStaging{};
		if (!rawData.positions.empty()) {
			posBufferGPU = vk::Buffer::createBuffer(
				allocator,
				rawData.positions.size() * sizeof(glm::vec3),
				VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				0,
				VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
			);
			posBufferStaging = vk::Buffer::createBuffer(
				allocator,
				rawData.positions.size() * sizeof(glm::vec3),
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
			);

			mapToGPU(allocator, posBufferGPU, posBufferStaging, rawData.positions);
		}

		// Texcoords
		vk::Buffer texBufferGPU{};
		vk::Buffer texBufferStaging{};
		if (!rawData.texcoords.empty()) {
			texBufferGPU = vk::Buffer::createBuffer(
				allocator,
				rawData.texcoords.size() * sizeof(glm::vec2),
				VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				0,
				VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
			);
			texBufferStaging = vk::Buffer::createBuffer(
				allocator,
				rawData.texcoords.size() * sizeof(glm::vec2),
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
			);

			mapToGPU(allocator, texBufferGPU, texBufferStaging, rawData.texcoords);
		}

		// Normals
		vk::Buffer normBufferGPU{};
		vk::Buffer normBufferStaging{};
		if (!rawData.normals.empty()) {
			normBufferGPU = vk::Buffer::createBuffer(
				allocator,
				rawData.normals.size() * sizeof(glm::vec3),
				VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				0,
				VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
			);
			normBufferStaging = vk::Buffer::createBuffer(
				allocator,
				rawData.normals.size() * sizeof(glm::vec3),
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
			);

			mapToGPU(allocator, normBufferGPU, normBufferStaging, rawData.normals);
		}
		
		// TBN frame
		vk::Buffer TBNBufferGPU{};
		vk::Buffer TBNBufferStaging{};
		if (!rawData.TBNs.empty()) {
			TBNBufferGPU = vk::Buffer::createBuffer(
				allocator,
				rawData.TBNs.size() * sizeof(uint32_t),
				VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				0,
				VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
			);
			TBNBufferStaging = vk::Buffer::createBuffer(
				allocator,
				rawData.TBNs.size() * sizeof(uint32_t),
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
			);

			mapToGPU(allocator, TBNBufferGPU, TBNBufferStaging, rawData.TBNs);
		}

		// Indices
		vk::Buffer indexBufferGPU{};
		vk::Buffer indexBufferStaging{};
		if (!rawData.indices.empty()) {
			indexBufferGPU = vk::Buffer::createBuffer(
				allocator,
				rawData.indices.size() * sizeof(uint32_t),
				VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				0,
				VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
			);
			indexBufferStaging = vk::Buffer::createBuffer(
				allocator,
				rawData.indices.size() * sizeof(uint32_t),
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
			);

			mapToGPU(allocator, indexBufferGPU, indexBufferStaging, rawData.indices);
		}

		// Upload to GPU
		VkCommandBuffer uploadCmdBuf = VkUtils::createCommandBuffer(window, window.getDevice()->getCmdPool());
		VkUtils::beginCommandBuffer(uploadCmdBuf);

		// If a vector is empty, the copy method does nothing
		copyToGPU(uploadCmdBuf, posBufferGPU, posBufferStaging, rawData.positions);
		copyToGPU(uploadCmdBuf, texBufferGPU, texBufferStaging, rawData.texcoords);
		copyToGPU(uploadCmdBuf, normBufferGPU, normBufferStaging, rawData.normals);
		copyToGPU(uploadCmdBuf, TBNBufferGPU, TBNBufferStaging, rawData.TBNs);
		copyToGPU(uploadCmdBuf, indexBufferGPU, indexBufferStaging, rawData.indices);

		VkUtils::endAndSubmitCommandBuffer(window, uploadCmdBuf);

		// Move into meshData
		meshData.posBuffer = std::move(posBufferGPU);
		meshData.texCoordBuffer = std::move(texBufferGPU);
		meshData.normalsBuffer = std::move(normBufferGPU);
		meshData.tbnFrameBuffer = std::move(TBNBufferGPU);
		meshData.indicesBuffer = std::move(indexBufferGPU);
		meshData.indicesCount = rawData.indices.size();

		return meshData;
	}

}