#include "Textures.hpp"

#include "error.hpp"

#include <unordered_map>

namespace Textures {

	static std::unordered_map<Texture, TextureBuffer> textureBuffers;

	void initialise() {
		registerTexture(Texture::MAIN_HDR, TextureBuffer::Builder::get()
			->withDescription(TextureDescs::HDR)
			->build()
		);
	}

	void registerTexture(Texture key, TextureBuffer textureBuffer) {

	}

	TextureBuffer& get(Texture texture) {
		try {
			return textureBuffers.at(texture);
		} catch (const std::out_of_range& e) {
			throw Utils::Error("Textures: Could not fine texture '%d'! Maybe it has not been initialised?\n", texture);
		}
	}

	bool isOfDepthFormat(ImageFormat format) {
		switch (format) {
			case ImageFormat::D32: 
				return true;
			default: 
				return false;
		}
	}

	bool isOfColorFormat(ImageFormat format) {
		switch (format) {
			case ImageFormat::RGBA16:
			case ImageFormat::RGBA8:
			case ImageFormat::B10GR11_PACK32:
			case ImageFormat::R8:
			case ImageFormat::RG32:
			case ImageFormat::RG16_SNORM:
				return true;
			default: 
				return false;
		}
	}

	bool isOfDepthLayout(ImageLayout layout) {
		switch (layout) {
			case ImageLayout::DEPTH:
			case ImageLayout::DEPTH_READ_ONLY:
			case ImageLayout::DEPTH_STENCIL:
			case ImageLayout::DEPTH_STENCIL_READ_ONLY:
				return true;
			default:
				return false;
		}
	}

	bool isOfColorLayout(ImageLayout layout) {
		switch (layout) {
			case ImageLayout::COLOR:
				return true;
			default:
				return false;
		}
	}

}