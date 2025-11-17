#pragma once

#include "../vulkan/objects/VkBuffer.hpp"

#include <string>
#include <vector>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

struct RawMeshData {
	std::string name;
	std::vector<glm::vec3> positions;
	std::vector<glm::vec2> texcoords;
	std::vector<glm::vec3> normals;
	std::vector<uint32_t> TBNs;
	std::vector<uint32_t> indices;
};

struct MeshData {
	vk::Buffer posBuffer{};
	vk::Buffer texCoordBuffer{};
	vk::Buffer normalsBuffer{};
	vk::Buffer tbnFrameBuffer{};
	vk::Buffer indicesBuffer{};
	std::size_t indicesCount = 0;
	std::uint32_t materialId = 0;
	bool hasAlphaMask = false;
};

struct LineMeshData {
	vk::Buffer posBuffer{};
	vk::Buffer colBuffer{};
	vk::Buffer indicesBuffer{};
	vk::Buffer posBufferStaging{};
	vk::Buffer colBufferStaging{};
	vk::Buffer indicesBufferStaging{};
	std::size_t indicesCount = 0;
};