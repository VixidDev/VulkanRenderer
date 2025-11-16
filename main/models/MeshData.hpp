#pragma once

#include "../vulkan/objects/VkBuffer.hpp"

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