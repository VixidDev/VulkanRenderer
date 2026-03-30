#include "Textures.hpp"

#include "../rendering/lights/Lights.hpp"

#include "error.hpp"

#include <unordered_map>

namespace Textures {

	static std::unordered_map<Texture, TextureBuffer> textureBuffers;

	void initialise() {
		registerTexture(Texture::MAIN_HDR, TextureBuffer::Builder::get()
			->withDescription(TextureDescs::HDR)
			// Loaded as attachment in main pass
			// Sampled in tonemap
			->hasFutureUse(TextureUse::ATTACHMENT_LOAD | TextureUse::TEXTURE_SAMPLE)
			->build()
		);
		registerTexture(Texture::MAIN_SDR, TextureBuffer::Builder::get()
			->withDescription(TextureDescs::SDR)
			// Sampled by FXAA/Mosaic/Any post process after tonemapping
			->hasFutureUse(TextureUse::TEXTURE_SAMPLE)
			->build()
		);
		registerTexture(Texture::INTERMEDIATE_HDR, TextureBuffer::Builder::get()
			->withDescription(TextureDescs::HDR)
			// Could be loaded as attachment in tonemap
			// Could be sampled in tonemap
			->hasFutureUse(TextureUse::ATTACHMENT_LOAD | TextureUse::TEXTURE_SAMPLE)
			->build()
		);
		registerTexture(Texture::INTERMEDIATE_SDR, TextureBuffer::Builder::get()
			->withDescription(TextureDescs::SDR)
			// Could be loaded as attachment in post processing
			// Could be sampled in post processing
			->hasFutureUse(TextureUse::ATTACHMENT_LOAD | TextureUse::TEXTURE_SAMPLE)
			->build()
		);
		registerTexture(Texture::DEPTH, TextureBuffer::Builder::get()
			->withDescription(TextureDescs::DEPTH)
			// Sampled by SSAO step
			->hasFutureUse(TextureUse::TEXTURE_SAMPLE)
			->build()
		);
		registerTexture(Texture::BRIGHTNESS, TextureBuffer::Builder::get()
			->withDescription(TextureDescs::HDR)
			// Sampled by bloom step
			->hasFutureUse(TextureUse::TEXTURE_SAMPLE)
			->build()
		);
		// Used for packing normals
		registerTexture(Texture::GBUFFER1, TextureBuffer::Builder::get()
			->withDescription(TextureDescs::B10GR11_PACK32)
			// Sampled by deferred shading step
			->hasFutureUse(TextureUse::TEXTURE_SAMPLE)
			->build()
		);
		// Used for albedo (rgb) and roughness (a) packing
		registerTexture(Texture::GBUFFER2, TextureBuffer::Builder::get()
			->withDescription(TextureDescs::SDR)
			// Sampled by deferred shading step
			->hasFutureUse(TextureUse::TEXTURE_SAMPLE)
			->build()
		);
		// Used for emissive (rgb) and metalness (a) packing
		registerTexture(Texture::GBUFFER3, TextureBuffer::Builder::get()
			->withDescription(TextureDescs::SDR)
			// Sampled by deferred shading step
			->hasFutureUse(TextureUse::TEXTURE_SAMPLE)
			->build()
		);
		registerTexture(Texture::BLUR_OUTPUT, TextureBuffer::Builder::get()
			->withDescription(TextureDescs::HDR)
			// Sampled when compositing with main render target
			->hasFutureUse(TextureUse::TEXTURE_SAMPLE)
			->build()
		);
		registerTexture(Texture::NOISE, TextureBuffer::Builder::get()
			->withDescription(TextureDescs::NOISE)
			// Sampled in SSAO
			->hasFutureUse(TextureUse::TEXTURE_SAMPLE)
			->build()
		);
		registerTexture(Texture::SSAO, TextureBuffer::Builder::get()
			->withDescription(TextureDescs::R8)
			->withExtent(ExtentRatio::HALF_SWAPCHAIN)
			// Sampled in SSAO H-blur
			->hasFutureUse(TextureUse::TEXTURE_SAMPLE)
			->build()
		);
		registerTexture(Texture::SSAO_HBLUR, TextureBuffer::Builder::get()
			->withDescription(TextureDescs::R8)
			// Sampled in SSAO V-blur
			->hasFutureUse(TextureUse::TEXTURE_SAMPLE)
			->build()
		);
		registerTexture(Texture::SSAO_VBLUR, TextureBuffer::Builder::get()
			->withDescription(TextureDescs::R8)
			// Sampled in main shading stage
			->hasFutureUse(TextureUse::TEXTURE_SAMPLE)
			->build()
		);
		registerTexture(Texture::SKYBOX, TextureBuffer::Builder::get()
			->withDescription(TextureDescs::SKYBOX)
			->withFlags(ImageCreate::CUBE_COMPATIBLE)
			//->withExtent(ExtentRatio::CUSTOM)
			->withViewType(ImageViewType::TYPE_CUBE)
			// Sampled in sun and main shading stage
			->hasFutureUse(TextureUse::TEXTURE_SAMPLE)
			->build()
		);
	}

	// Shadow texture buffers depend on lights in the scene for
	// their definition and counts
	void initialiseDeferredTextures() {
		std::size_t pointLights = Lights::getNbShadowPointLights();
		std::size_t directionalLights = Lights::getNbShadowDirectionalLights();
		std::size_t spotLights = Lights::getNbShadowSpotLights();

		// Point light shadow maps
		registerTexture(Texture::SHADOW_POINT, TextureBuffer::Builder::get()
			->withDescription(TextureDescs::DEPTH)
			->withFlags(ImageCreate::CUBE_COMPATIBLE)
			->withArrayLayers(pointLights)
			->withViewType(ImageViewType::TYPE_CUBE_ARRAY)
			->hasFutureUse(TextureUse::TEXTURE_SAMPLE)
			->isRenderTarget()
			->build()
		);
		registerTexture(Texture::SHADOW_POINT_VSM, TextureBuffer::Builder::get()
			->withDescription(TextureDescs::RG32)
			->withFlags(ImageCreate::CUBE_COMPATIBLE)
			->withArrayLayers(pointLights)
			->withViewType(ImageViewType::TYPE_CUBE_ARRAY)
			->hasFutureUse(TextureUse::TEXTURE_SAMPLE)
			->isRenderTarget()
			->build()
		);
		registerTexture(Texture::SHADOW_POINT_VSM_DEPTH, TextureBuffer::Builder::get()
			->withDescription(TextureDescs::DEPTH)
			->withFlags(ImageCreate::CUBE_COMPATIBLE)
			->withArrayLayers(pointLights)
			->withViewType(ImageViewType::TYPE_CUBE_ARRAY)
			->hasFutureUse(TextureUse::TEXTURE_SAMPLE)
			->isRenderTarget()
			->build()
		);
		registerTexture(Texture::SHADOW_POINT_VSM_BLUR, TextureBuffer::Builder::get()
			->withDescription(TextureDescs::RG32)
			->withFlags(ImageCreate::CUBE_COMPATIBLE)
			->withArrayLayers(pointLights)
			->withViewType(ImageViewType::TYPE_CUBE_ARRAY)
			->hasFutureUse(TextureUse::TEXTURE_SAMPLE)
			->isRenderTarget()
			->build()
		);

		// Directional light shadow maps
		registerTexture(Texture::SHADOW_DIRECTIONAL, TextureBuffer::Builder::get()
			->withDescription(TextureDescs::DEPTH)
			->withArrayLayers(directionalLights)
			->hasFutureUse(TextureUse::TEXTURE_SAMPLE)
			->isRenderTarget()
			->build()
		);
		registerTexture(Texture::SHADOW_DIRECTIONAL_VSM, TextureBuffer::Builder::get()
			->withDescription(TextureDescs::RG32)
			->withArrayLayers(directionalLights)
			->hasFutureUse(TextureUse::TEXTURE_SAMPLE)
			->isRenderTarget()
			->build()
		);
		registerTexture(Texture::SHADOW_DIRECTIONAL_VSM_DEPTH, TextureBuffer::Builder::get()
			->withDescription(TextureDescs::DEPTH)
			->withArrayLayers(directionalLights)
			->hasFutureUse(TextureUse::TEXTURE_SAMPLE)
			->isRenderTarget()
			->build()
		);
		registerTexture(Texture::SHADOW_DIRECTIONAL_VSM_BLUR, TextureBuffer::Builder::get()
			->withDescription(TextureDescs::RG32)
			->withArrayLayers(directionalLights)
			->hasFutureUse(TextureUse::TEXTURE_SAMPLE)
			->isRenderTarget()
			->build()
		);

		// Spot light shadow maps
		registerTexture(Texture::SHADOW_SPOT, TextureBuffer::Builder::get()
			->withDescription(TextureDescs::DEPTH)
			->withArrayLayers(spotLights)
			->hasFutureUse(TextureUse::TEXTURE_SAMPLE)
			->isRenderTarget()
			->build()
		);
		registerTexture(Texture::SHADOW_SPOT_VSM, TextureBuffer::Builder::get()
			->withDescription(TextureDescs::RG32)
			->withArrayLayers(spotLights)
			->hasFutureUse(TextureUse::TEXTURE_SAMPLE)
			->isRenderTarget()
			->build()
		);
		registerTexture(Texture::SHADOW_SPOT_VSM_DEPTH, TextureBuffer::Builder::get()
			->withDescription(TextureDescs::DEPTH)
			->withArrayLayers(spotLights)
			->hasFutureUse(TextureUse::TEXTURE_SAMPLE)
			->isRenderTarget()
			->build()
		);
		registerTexture(Texture::SHADOW_SPOT_VSM_BLUR, TextureBuffer::Builder::get()
			->withDescription(TextureDescs::RG32)
			->withArrayLayers(spotLights)
			->hasFutureUse(TextureUse::TEXTURE_SAMPLE)
			->isRenderTarget()
			->build()
		);
	}

	void registerTexture(Texture key, TextureBuffer textureBuffer) {
		textureBuffers[key] = std::move(textureBuffer);
	}

	TextureBuffer& get(Texture texture) {
		try {
			return textureBuffers.at(texture);
		} catch (const std::out_of_range& e) {
			throw Utils::Error("Textures: Could not finf texture '%d'! Maybe it has not been initialised?\n", texture);
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