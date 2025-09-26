/*
* File credits to Markus Billeter
*/
#include <iterator>
#include <vector>
#include <typeinfo>
#include <exception>
#include <filesystem>
#include <system_error>
#include <unordered_map>

#include <cstdio>
#include <cstring>

#include <tgen.h>
#include <glm/glm.hpp>

#include "IndexMesh.hpp"
#include "InputModel.hpp"
#include "LoadModelObj.hpp"

#include "../utils/Error.hpp"

namespace {
	// constants
	/* File "magic". The first 16 bytes of our custom file are equal to this
	 * magic value. This allows us to check whether a certain file is
	 * (probably) of the right type.
	 *
	 * When picking a signature there are a few considerations. For example,
	 * including non-printable characters (e.g. the \0) early keeps the file
	 * from being misidentified as text.
	 */
	constexpr char kFileMagic[16] = "\0\0VixidVkMesh";

	// Note: change the file variant if file format changes!
	constexpr char kFileVariant[16] = "25-tan";

	// Fallback texture for RGBA 1111 and Grayscale 1
	constexpr char kTextureFallbackR1[] = "assets-src/main/r1.png";
	constexpr char kTextureFallbackRGBA1111[] = "assets-src/main/rgba1111.png";
	constexpr char kTextureFallbackRGB000[] = "assets-src/main/rgb000.png";

	// types
	struct TextureInfo_ {
		std::uint32_t uniqueId;
		std::uint8_t space;
		std::uint8_t channels;
		std::string newPath;
	};

	// local functions:
	void processModel(
		const char* output,
		const char* inputOBJ,
		const glm::mat4x4& staticTransform = glm::mat4x4(1.f));

	InputModel normalize(InputModel inputModel);

	void writeModelData(
		FILE* out,
		const InputModel& model,
		const std::vector<IndexedMesh>& indexedMeshes,
		const std::unordered_map<std::string,TextureInfo_>& textures);

	std::vector<IndexedMesh> indexMeshes(
		const InputModel& model,
		float errorTolerance = 1e-5f);

	std::unordered_map<std::string,TextureInfo_> findUniqueTextures(const InputModel& model);

	std::unordered_map<std::string,TextureInfo_> newPaths(
		std::unordered_map<std::string,TextureInfo_> textures,
		const std::filesystem::path& texDir);

}


int main() try {
#if !defined(NDEBUG)
	std::printf("Suggest running this in release mode (it appears to be running in debug)\n");
	std::printf("Especially under VisualStudio/MSVC, the debug build seems very slow.\n");
	/* A few notes:
	 * 
	 * I have not profiled this at all. The following are based on previous
	 * experience(s).
	 *
	 * - ZStd benefits immensely from compiler optimizations.
	 * 
	 * - Under MSVC, std::unordered_set performs quite badly in debug mode. This
	 *   may be further related to other debug-related options (e.g., extended
	 *   iterator checking...).
	 * 
	 *   Normally, I avoid unordered_set here, and instead rely on one of the many
	 *   high quality flat_set implementations. They tend to be faster from the 
	 *   get go and perform more equally under different compilers. 
	 * 
	 * - NDEBUG is the standard macro to control the behaviour of assert(). When 
	 *   NDEBUG is defined, assert() will "do nothing" (they're expanded to an 
	 *   empty statement). This is typically desirable in a release build, but not
	 *   necessary or guaranteed. (Indeed, the premake sets NDEBUG explicitly for
	 *   this project -- this is why the check above works. But don't rely on this
	 *   blindly.)
	 * 
	 * - The VisualStudio interactive debugger's heap profiler (the thing that 
	 *   shows you the memory usage graph) carries a measurable overhead as well.
	 *
	 * The binary .vixidvkmesh should be unchanged between debug and release
	 * builds, so you can safely use the release build to create the file once,
	 * even while debugging the main A12 program.
	 */
#endif

	processModel(
		"assets/main/suntemple.vixidvkmesh",
		"assets-src/main/suntemple.obj-zstd");

	return 0;
} catch (const std::exception& error) {
	std::fprintf(stderr, "Top-level exception [%s]:\n%s\nBye.\n", typeid(error).name(), error.what());
	return 1;
}

namespace {
	void processModel(const char* output, const char* inputOBJ, const glm::mat4x4& staticTransform) {
		static constexpr std::size_t vertexSize = sizeof(float) * (3 + 3 + 2);

		// Figure out output paths
		const std::filesystem::path outname(output);
		const std::filesystem::path rootdir = outname.parent_path();
		const std::filesystem::path basename = outname.stem();
		const std::filesystem::path texdir = basename.string() + "-tex";

		// Load input model
		const InputModel model = normalize(loadCompressedWavefrontObj(inputOBJ));

		std::size_t inputVerts = 0;
		for (const InputMeshInfo& imesh : model.meshes)
			inputVerts += imesh.vertexCount;

		std::printf("%s: %zu meshes, %zu materials\n", inputOBJ, model.meshes.size(), model.materials.size());
		std::printf(" - triangle soup vertices: %zu => %zu kB\n", inputVerts, inputVerts * vertexSize / 1024);

		// Index meshes
		const auto indexed = indexMeshes(model);

		std::size_t outputVerts = 0, outputIndices = 0;
		for (const auto& mesh : indexed) {
			outputVerts += mesh.vert.size();
			outputIndices += mesh.indices.size();
		}

		std::printf(" - indexed vertices: %zu with %zu indices => %zu kB\n", outputVerts, outputIndices, (outputVerts*vertexSize + outputIndices*sizeof(std::uint32_t))/1024);

		// Find list of unique textures
		const auto textures = newPaths(findUniqueTextures(model), texdir);

		std::printf(" - unique textures: %zu\n", textures.size());

		// Ensure output directory exists
		std::filesystem::create_directories(rootdir);

		// Output mesh data
		auto mainpath = rootdir / basename;
		mainpath.replace_extension("vixidvkmesh");

		FILE* fof = std::fopen(mainpath.string().c_str(), "wb");
		if (!fof)
			throw Utils::Error("Unable to open '%s' for writing", mainpath.string().c_str());

		try {
			writeModelData(fof, model, indexed, textures);
		} catch (...) {
			std::fclose(fof);
			throw;
		}

		std::fclose(fof);

		// Copy textures
		std::filesystem::create_directories(rootdir / texdir);

		std::size_t errors = 0;
		for (const auto& entry : textures) {
			const auto dest = rootdir / entry.second.newPath;

			std::error_code ec;
			bool ret = std::filesystem::copy_file( 
				entry.first,
				dest,
				std::filesystem::copy_options::none,
				ec);

			if (!ret) {
				++errors;
				std::fprintf(stderr, "copy_file(): '%s' failed: %s (%s)\n", dest.string().c_str(), ec.message().c_str(), ec.category().name());
			}
		}

		const auto total = textures.size();
		std::printf("Copied %zu textures out of %zu.\n", total - errors, total);
		if (errors) {
			std::fprintf(stderr, "Some copies reported an error. Currently, the code will never overwrite existing files. The errors likely just indicate that the file was copied previously. Remove old files manually, if necessary.\n");
		}
	}

	InputModel normalize(InputModel inputModel) {
		for (auto& mat : inputModel.materials) {
			if (mat.baseColorTexturePath.empty())
				mat.baseColorTexturePath = kTextureFallbackRGBA1111;
			if (mat.roughnessTexturePath.empty())
				mat.roughnessTexturePath = kTextureFallbackR1;
			if (mat.metalnessTexturePath.empty())
				mat.metalnessTexturePath = kTextureFallbackR1;
			if (mat.emissiveTexturePath.empty())
				mat.emissiveTexturePath = kTextureFallbackRGB000;
		}

		return inputModel; // This should use the move constructor implicitly.
	}

	void checkedWrite(FILE* out, std::size_t bytes, const void* data) {
		const auto ret = std::fwrite(data, 1, bytes, out);

		if (ret != bytes)
			throw Utils::Error( "fwrite() failed: %zu instead of %zu", ret, bytes);
	}

	void writeString(FILE* out, const char* string) {
		// Write a string
		// Format:
		//  - uint32_t : N = length of string in bytes, including terminating '\0'
		//  - N x char : string
		const std::uint32_t length = std::uint32_t(std::strlen(string) + 1);
		checkedWrite(out, sizeof(std::uint32_t), &length);

		checkedWrite(out, length, string);
	}

	void writeModelData(
		FILE* out, 
		const InputModel& model, 
		const std::vector<IndexedMesh>& indexedMeshes, 
		const std::unordered_map<std::string,TextureInfo_>& textures) 
	{
		// Write header
		// Format:
		//   - char[16] : file magic
		//   - char[16] : file variant ID
		checkedWrite(out, sizeof(char) * 16, kFileMagic);
		checkedWrite(out, sizeof(char) * 16, kFileVariant);
		
		// Write list of unique textures
		// Format:
		//  - unit32_t : U = number of unique textures
		//  - repeat U times:
		//    - string : path to texture 
		//    - uint8_t : texture color space (0 = unorm, 1 = srgb)
		//    - uint8_t : number of channels in texture
		std::vector<const TextureInfo_*> orderedUnqiue(textures.size());
		for (const auto& tex : textures) {
			assert(!orderedUnqiue[tex.second.uniqueId]);
			orderedUnqiue[tex.second.uniqueId] = &tex.second;
		}

		const std::uint32_t textureCount = std::uint32_t(orderedUnqiue.size());
		checkedWrite(out, sizeof(textureCount), &textureCount);

		for (const auto& tex : orderedUnqiue) {
			assert(tex);
			writeString(out, tex->newPath.c_str());

			std::uint8_t space = tex->space;
			checkedWrite(out, sizeof(space), &space);

			std::uint8_t channels = tex->channels;
			checkedWrite(out, sizeof(channels), &channels);
		}

		// Write material information
		// Format:
		//  - uint32_t : M = number of materials
		//  - repeat M times:
		//    - uin32_t : base color texture index
		//    - uin32_t : roughness texture index
		//    - uin32_t : metalness texture index
		//    - uin32_t : alphaMask texture index (or 0xffffffff if none)
		//    - uin32_t : normalMap texture index (or 0xffffffff if none)
		//    - uin32_t : emissive texture index
		const std::uint32_t materialCount = std::uint32_t(model.materials.size());
		checkedWrite(out, sizeof(materialCount), &materialCount);

		for (const auto& mat : model.materials) {
			const auto write_tex_ = [&] (const std::string& texturePath) {
				if (texturePath.empty()) {
					static constexpr std::uint32_t sentinel = ~std::uint32_t(0);
					checkedWrite(out, sizeof(std::uint32_t), &sentinel);
					return;
				}

				const auto it = textures.find(texturePath);
				assert(textures.end() != it);

				checkedWrite(out, sizeof(std::uint32_t), &it->second.uniqueId);
			};

			write_tex_(mat.baseColorTexturePath);
			write_tex_(mat.roughnessTexturePath);
			write_tex_(mat.metalnessTexturePath);
			write_tex_(mat.alphaMaskTexturePath);
			write_tex_(mat.normalMapTexturePath);
			write_tex_(mat.emissiveTexturePath);
		}

		// Write mesh data
		// Format:
		//  - uint32_t : M = number of meshes
		//  - repeat M times:
		//    - uint32_t : material index
		//    - uint32_t : V = number of vertices
		//    - uint32_t : I = number of indices
		//    - repeat V times: vec3 position
		//    - repeat V times: vec3 normal
		//    - repeat V times: vec2 texture coordinate
		//    - repeat V times: vec4 tangent
		//    - repeat I times: uint32_t index
		const std::uint32_t meshCount = std::uint32_t(model.meshes.size());
		checkedWrite(out, sizeof(meshCount), &meshCount);

		assert(model.meshes.size() == indexedMeshes.size());
		for (std::size_t i = 0; i < model.meshes.size(); ++i) {
			const auto& mmesh = model.meshes[i];

			std::uint32_t materialIndex = std::uint32_t(mmesh.materialIndex);
			checkedWrite(out, sizeof(materialIndex), &materialIndex);

			auto const& imesh = indexedMeshes[i];

			std::uint32_t vertexCount = std::uint32_t(imesh.vert.size());
			checkedWrite(out, sizeof(vertexCount), &vertexCount);
			std::uint32_t indexCount = std::uint32_t(imesh.indices.size());
			checkedWrite(out, sizeof(indexCount), &indexCount);

			checkedWrite(out, sizeof(glm::vec3) * vertexCount, imesh.vert.data());
			checkedWrite(out, sizeof(glm::vec3) * vertexCount, imesh.norm.data());
			checkedWrite(out, sizeof(glm::vec2) * vertexCount, imesh.text.data());
			checkedWrite(out, sizeof(glm::vec4) * vertexCount, imesh.tangent.data()); // imesh.tangent populated in IndexMesh.hpp
			checkedWrite(out, sizeof(std::uint32_t) * vertexCount, imesh.tangentComp.data());

			checkedWrite(out, sizeof(std::uint32_t) * indexCount, imesh.indices.data());
		}
	}

	std::vector<IndexedMesh> indexMeshes(const InputModel& model, float errorTolerance) {
		std::vector<IndexedMesh> indexed;

		for (const auto& imesh : model.meshes) {
			const auto endIndex = imesh.vertexStartIndex + imesh.vertexCount;

			TriangleSoup soup;

			soup.vert.reserve(imesh.vertexCount);
			for (std::size_t i = imesh.vertexStartIndex; i < endIndex; ++i)
				soup.vert.emplace_back(model.positions[i]);

			soup.text.reserve(imesh.vertexCount);
			for (std::size_t i = imesh.vertexStartIndex; i < endIndex; ++i)
				soup.text.emplace_back(model.texcoords[i]);

			soup.norm.reserve(imesh.vertexCount);
			for (std::size_t i = imesh.vertexStartIndex; i < endIndex; ++i)
				soup.norm.emplace_back(model.normals[i]);

			indexed.emplace_back(makeIndexedMesh(soup, errorTolerance));
		}

		return indexed;
	}

	std::unordered_map<std::string,TextureInfo_> findUniqueTextures(const InputModel& model) {
		std::unordered_map<std::string,TextureInfo_> unique;

		std::uint32_t texid = 0;
		const auto add_unique_ = [&] (const std::string& path, std::uint8_t space, std::uint8_t channels) {
			if (path.empty())
				return;

			TextureInfo_ info{};
			info.uniqueId = texid;
			info.space = space;
			info.channels = channels;

			auto const [it, isNew] = unique.emplace(std::make_pair(path,info));

			if (isNew)
				++texid;
		};

		for (const auto& mat : model.materials) {
			add_unique_(mat.baseColorTexturePath, 1, 4);
			add_unique_(mat.roughnessTexturePath, 0, 1); 
			add_unique_(mat.metalnessTexturePath, 0, 1); 
			add_unique_(mat.alphaMaskTexturePath, 1, 4);  // assume == baseColor
			add_unique_(mat.normalMapTexturePath, 0, 3);  // xyz only
			add_unique_(mat.emissiveTexturePath, 1, 4); 
		}

		return unique;
	}

	std::unordered_map<std::string,TextureInfo_> newPaths(
		std::unordered_map<std::string,TextureInfo_> textures, 
		const std::filesystem::path& texDir)
	{
		for (auto& entry : textures) {
			const std::filesystem::path originalPath(entry.first);
			const auto filename = originalPath.filename();
			const auto newpath = texDir / filename;
		
			auto& info = entry.second;
			info.newPath = newpath.string();
		}

		// Note: 'textures' is still local to the function, so there is no need
		// to explicitly std::move() it. However, since it is passed in as an
		// argument, NRVO is unlikely to occur.
		return textures; 
	}
}



