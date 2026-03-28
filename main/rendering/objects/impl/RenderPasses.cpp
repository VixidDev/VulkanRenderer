#include "RenderPasses.hpp"

#include "Textures.hpp"

#include <unordered_map>

namespace RenderPasses {

	static std::unordered_map<Pass, RenderPass> renderPasses;

	void initialise() {
		registerPass(Pass::SHADOW, RenderPass::Builder::get()
			->withDepthAttachment(Texture::SHADOW_SPOT, AttachmentLoadOp::CLEAR, AttachmentStoreOp::STORE, ImageLayout::DEPTH_STENCIL_READ_ONLY)
			->build()
		);

		// Forward Pass
		// Writes to main color, brightness color, depth buffer
		// Reads from main color, shadow maps (depth buffers), and SSAO texture (color texture)
		//
		// This render pass must wait until previous passes are done with color writes (skybox writes to main output
		// and possibly SSAO texture) and depth writes (shadow maps (1st frame)), before being able to read/write to
		// color and depth
		// Future render passes must wait until this render pass is done with both color and depth writes before
		// reading/writing to them
		registerPass(Pass::FORWARD, RenderPass::Builder::get()
			// Main output image
			->withColourAttachment(Texture::MAIN_HDR, AttachmentLoadOp::LOAD, AttachmentStoreOp::STORE, ImageLayout::COLOR, ImageLayout::COLOR)
			// Brightness image
			->withColourAttachment(Texture::BRIGHTNESS, AttachmentLoadOp::CLEAR, AttachmentStoreOp::STORE, ImageLayout::COLOR)
			// Depth buffer
			->withDepthAttachment(Texture::DEPTH, AttachmentLoadOp::CLEAR, AttachmentStoreOp::DONT_CARE, ImageLayout::DEPTH)
			->usesDescriptorInShader(ImageType::COLOR) // Uses SSAO texture in shader
			->build()
		);
	}

	void registerPass(Pass key, RenderPass renderPass) {
		renderPasses[key] = std::move(renderPass);
	}

}