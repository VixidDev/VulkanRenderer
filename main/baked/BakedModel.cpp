#include "BakedModel.hpp"

#include <cstdio>
#include <cstring>

#include "Error.hpp"

namespace {
	constexpr char kFileMagic[16] = "\0\0VixidVkMesh";
	constexpr char kFileVariant[16] = "25-tan";

	constexpr std::uint32_t kMaxString = 32*1024;

	// functions
	BakedModel loadBakedModel_(FILE*, const char*);
}

BakedModel loadBakedModel(const char* modelPath) {
	FILE* fin = std::fopen(modelPath, "rb");
	if (!fin)
		throw Utils::Error("loadBakedModel(): unable to open '%s' for reading", modelPath);

	try {
		BakedModel ret = loadBakedModel_(fin, modelPath);
		std::fclose(fin);
		return ret;
	} catch (...) {
		std::fclose(fin);
		throw;
	}
}

namespace {
	void checkedRead_(FILE* fin, std::size_t bytes, void* buffer) {
		auto ret = std::fread(buffer, 1, bytes, fin);

		if (bytes != ret)
			throw Utils::Error("checkedRead_(): expected %zu bytes, got %zu", bytes, ret);
	}

	std::uint32_t readUint32_(FILE* fin) {
		std::uint32_t ret;
		checkedRead_(fin, sizeof(std::uint32_t), &ret);
		return ret;
	}

	std::string readString_(FILE* fin) {
		const std::uint32_t length = readUint32_(fin);

		if (length >= kMaxString)
			throw Utils::Error("readString_(): unexpectedly long string (%u bytes)", length);

		std::string ret;
		ret.resize(length);

		checkedRead_(fin, length, ret.data());
		return ret;
	}

	BakedModel loadBakedModel_(FILE* fin, const char* inputName) {
		BakedModel ret;

		// Figure out base path
		const char* pathBeg = inputName;
		const char* pathEnd = std::strrchr(pathBeg, '/');
	
		const std::string prefix = pathEnd ? std::string(pathBeg, pathEnd+1) : "";

		// Read header and verify file magic and variant
		char magic[16];
		checkedRead_(fin, 16, magic);

		if (0 != std::memcmp(magic, kFileMagic, 16))
			throw Utils::Error("loadBakedModel_(): %s: invalid file signature!", inputName);

		char variant[16];
		checkedRead_(fin, 16, variant);

		if (0 != std::memcmp(variant, kFileVariant, 16))
			throw Utils::Error("loadBakedModel_(): %s: file variant is '%s', expected '%s'", inputName, variant, kFileVariant);

		// Read texture info
		const std::uint32_t textureCount = readUint32_(fin);
		for (std::uint32_t i = 0; i < textureCount; ++i) {
			BakedTextureInfo info;
			info.path = prefix + readString_(fin);

			std::uint8_t space;
			checkedRead_(fin, sizeof(std::uint8_t), &space);
			info.space = ETextureSpace(space);

			std::uint8_t channels;
			checkedRead_(fin, sizeof(std::uint8_t), &channels);
			info.channels = channels;

			ret.textures.emplace_back(std::move(info));
		}

		// Read material info
		const std::uint32_t materialCount = readUint32_(fin);
		for (std::uint32_t i = 0; i < materialCount; ++i) {
			BakedMaterialInfo info;
			info.baseColorTextureId = readUint32_(fin);
			info.roughnessTextureId = readUint32_(fin);
			info.metalnessTextureId = readUint32_(fin);
			info.alphaMaskTextureId = readUint32_(fin);
			info.normalMapTextureId = readUint32_(fin);
			info.emissiveTextureId = readUint32_(fin);

			assert(info.baseColorTextureId < ret.textures.size());
			assert(info.roughnessTextureId < ret.textures.size());
			assert(info.metalnessTextureId < ret.textures.size());
			assert(info.emissiveTextureId < ret.textures.size());

			ret.materials.emplace_back(std::move(info));
		}

		// Read mesh data
		const std::uint32_t meshCount = readUint32_(fin);
		for (std::uint32_t i = 0; i < meshCount; ++i) {
			BakedMeshData data;
			data.materialId = readUint32_(fin);
			assert( data.materialId < ret.materials.size() );

			const std::uint32_t V = readUint32_(fin);
			const std::uint32_t I = readUint32_(fin);

			data.positions.resize(V);
			checkedRead_(fin, V * sizeof(glm::vec3), data.positions.data());

			data.normals.resize(V);
			checkedRead_(fin, V * sizeof(glm::vec3), data.normals.data());

			data.texcoords.resize(V);
			checkedRead_(fin, V * sizeof(glm::vec2), data.texcoords.data());

			data.tangents.resize(V);
			checkedRead_(fin, V * sizeof(glm::vec4), data.tangents.data());

			data.tangentsComp.resize(V);
			checkedRead_(fin, V * sizeof(std::uint32_t), data.tangentsComp.data());

			data.indices.resize(I);
			checkedRead_(fin, I * sizeof(std::uint32_t), data.indices.data());

			ret.meshes.emplace_back(std::move(data));
		}

		// Check
		char byte;
		const std::size_t check = std::fread(&byte, 1, 1, fin);
		
		if (0 != check)
			std::fprintf(stderr, "Note: '%s' contains trailing bytes\n", inputName);

		return ret;
	}
}
