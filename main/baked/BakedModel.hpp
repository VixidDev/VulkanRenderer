#pragma once

#include "../vulkan/objects/VkBuffer.hpp"

#include <string>
#include <vector>
#include <cstdint>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

/* Baked file format:
 *
 *  1. Header:
 *    - 16*char: file magic = "\0\0VixidVkMesh"
 *    - 16*char: variant = "25-tan" (maybe something different if changed in main-bake/main.cpp)
 *
 *  2. Textures
 *    - 1*uint32_t: U = number of (unique) textures
 *    - repeat U times:
 *      - string: path to texture
 *      - 1*uint8_t: texture color space (see ETextureSpace)
 *      - 1*uint8_t: number of channels in texture
 *
 *  3. Material information
 *    - 1*uint32_t: M = number of materials
 *    - repeat M times:
 *      - uint32_t: base color texture index
 *      - uint32_t: roughness texture index
 *      - uint32_t: metalness texture index
 *      - uint32_t: alpha mask texture index; set to 0xffffffff if not available
 *      - uint32_t: normal map texture index; set to 0xffffffff if not available
 *      - uint32_t: emissive texture index; set to 0xffffffff if not available
 *
 *  4. Mesh data
 *    - 1*uint32_t: M = number of meshes
 *    - repeat M times:
 *      - uint32_t : material index
 *      - uint32_t : V = number of vertices
 *      - uint32_t : I = number of indices
 *      - repeat V times: vec3 position
 *      - repeat V times: vec3 normal
 *      - repeat V times: vec2 texture coordinate
 *      - repeat V times: vec2 tangents
 *      - repeat V times: vec2 optimised tangents
 *      - repeat I times: uint32_t index
 *
 * Strings are stored as
 *   - 1*uint32_t: N = length of string in chars, including terminating \0
 *   - repeat N times: char in string
 *
 */

enum class ETextureSpace : std::uint8_t {
	unorm = 0,
	srgb = 1
};

struct BakedTextureInfo {
	std::string path;
	ETextureSpace space;
	std::uint8_t channels;
};

struct BakedMaterialInfo {
	std::uint32_t baseColorTextureId;
	std::uint32_t roughnessTextureId;
	std::uint32_t metalnessTextureId;
	std::uint32_t alphaMaskTextureId; // May be set to 0xffffffff if no alpha mask
	std::uint32_t normalMapTextureId; // May be set to 0xffffffff if no normal map
	std::uint32_t emissiveTextureId;  // May be set to 0xffffffff if no emissive map
};

struct BakedMeshData {
	std::uint32_t materialId;

	std::vector<glm::vec3> positions;
	std::vector<glm::vec2> texcoords;
	std::vector<glm::vec3> normals;
	std::vector<glm::vec4> tangents;
	std::vector<std::uint32_t> tangentsComp;

	std::vector<std::uint32_t> indices;
};

struct BakedModel {
	std::vector<BakedTextureInfo> textures;
	std::vector<BakedMaterialInfo> materials;
	std::vector<BakedMeshData> meshes;
};

BakedModel loadBakedModel(const char* modelPath);

