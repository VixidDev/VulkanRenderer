#pragma once

#include "../base/structure/Textures.hpp"
#include "../base/TextureBuffer.hpp"

namespace TextureDescs {

	static TextureDesc HDR = {
		.format = ImageFormat::RGBA16,
		.usage = (ImageUsage)(ImageUsage::COLOR | ImageUsage::INPUT_ATTACHMENT | ImageUsage::SAMPLED | ImageUsage::TRANSFER_SRC)
	};

	static TextureDesc SDR = {
		.format = ImageFormat::RGBA8,
		.usage = (ImageUsage)(ImageUsage::COLOR | ImageUsage::INPUT_ATTACHMENT | ImageUsage::SAMPLED | ImageUsage::TRANSFER_SRC)
	};

	static TextureDesc DEPTH = {
		.format = ImageFormat::D32,
		.usage = (ImageUsage)(ImageUsage::DEPTH_STENCIL | ImageUsage::INPUT_ATTACHMENT | ImageUsage::SAMPLED),
		.aspect = ImageAspect::DEPTH
	};

	static TextureDesc B10GR11_PACK32 = {
		.format = ImageFormat::B10GR11_PACK32,
		.usage = (ImageUsage)(ImageUsage::COLOR | ImageUsage::INPUT_ATTACHMENT | ImageUsage::SAMPLED | ImageUsage::TRANSFER_SRC)
	};

	static TextureDesc R8 = {
		.format = ImageFormat::R8,
		.usage = (ImageUsage)(ImageUsage::COLOR | ImageUsage::INPUT_ATTACHMENT | ImageUsage::SAMPLED | ImageUsage::TRANSFER_SRC)
	};

	static TextureDesc NOISE = {
		.format = ImageFormat::RG16_SNORM,
		.usage = (ImageUsage)(ImageUsage::SAMPLED | ImageUsage::TRANSFER_DST | ImageUsage::TRANSFER_SRC)
	};

	static TextureDesc RG32 = {
		.format = ImageFormat::RG32,
		.usage = (ImageUsage)(ImageUsage::COLOR | ImageUsage::INPUT_ATTACHMENT | ImageUsage::SAMPLED | ImageUsage::TRANSFER_SRC),
	};

}

enum Texture {
	MAIN_HDR,
	MAIN_SDR,
	INTERMEDIATE_HDR,
	INTERMEDIATE_SDR,
	DEPTH,
	BRIGHTNESS,
	GBUFFER1,
	GBUFFER2,
	GBUFFER3,
	BLUR_OUTPUT,
	NOISE,
	SSAO,
	SSAO_HBLUR,
	SSAO_VBLUR,
	SKYBOX,

	// Shadow textures
	SHADOW_POINT,
	SHADOW_DIRECTIONAL,
	SHADOW_SPOT,
	SHADOW_POINT_VSM,
	SHADOW_POINT_VSM_DEPTH,
	SHADOW_POINT_VSM_BLUR,
	SHADOW_DIRECTION_VSM,
	SHADOW_DIRECTION_VSM_DEPTH,
	SHADOW_DIRECTION_VSM_BLUR,
	SHADOW_SPOT_VSM,
	SHADOW_SPOT_VSM_DEPTH,
	SHADOW_SPOT_VSM_BLUR
};

namespace Textures {

	void initialise();
	void registerTexture(Texture key, TextureBuffer textureBuffer);
	TextureBuffer& get(Texture texture);
	bool isOfDepthFormat(ImageFormat format);
	bool isOfColorFormat(ImageFormat format);
	bool isOfDepthLayout(ImageLayout layout);
	bool isOfColorLayout(ImageLayout layout);
}