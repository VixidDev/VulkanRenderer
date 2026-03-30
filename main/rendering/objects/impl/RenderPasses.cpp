#include "RenderPasses.hpp"

#include "Textures.hpp"
#include "error.hpp"

#include <unordered_map>

namespace RenderPasses {

	static std::unordered_map<Pass, RenderPass> renderPasses;

	void initialise() {
		// TODO: replace 1st argument for color and depth attachments with texture format only
		// and not the entire texture definition, since we don't need anything else but the
		// format to create the render pass
		registerPass(Pass::SHADOW, RenderPass::Builder::get()
			->withDepthAttachment(Texture::SHADOW_SPOT, LoadOp::CLEAR, StoreOp::STORE, ImageLayout::DEPTH_STENCIL_READ_ONLY)
			->build()
		);
		registerPass(Pass::SHADOW_VSM, RenderPass::Builder::get()
			->withColourAttachment(Texture::SHADOW_SPOT_VSM, LoadOp::CLEAR, StoreOp::STORE, ImageLayout::SHADER_READ_ONLY)
			->withDepthAttachment(Texture::SHADOW_SPOT_VSM_DEPTH, LoadOp::CLEAR, StoreOp::DONT_CARE, ImageLayout::DEPTH_STENCIL)
			->build()
		);
		registerPass(Pass::SHADOW_VSM_BLUR, RenderPass::Builder::get()
			->withColourAttachment(Texture::SHADOW_SPOT_VSM_BLUR, LoadOp::CLEAR, StoreOp::STORE, ImageLayout::SHADER_READ_ONLY)
			->usesDescriptorInShader(ImageType::COLOR) // Uses SSAO or SSAO H-blur step as sampler
			->build()
		);
		registerPass(Pass::SKYBOX, RenderPass::Builder::get()
			->withColourAttachment(Texture::SKYBOX, LoadOp::LOAD, StoreOp::STORE, ImageLayout::COLOR)
			->build()
		);
		registerPass(Pass::SUN, RenderPass::Builder::get()
			->withColourAttachment(Texture::MAIN_HDR, LoadOp::LOAD, StoreOp::STORE, ImageLayout::COLOR)
			->build()
		);
		registerPass(Pass::PRE_SSAO, RenderPass::Builder::get()
			->withColourAttachment(Texture::GBUFFER1, LoadOp::CLEAR, StoreOp::STORE, ImageLayout::SHADER_READ_ONLY)
			->withDepthAttachment(Texture::DEPTH, LoadOp::CLEAR, StoreOp::STORE, ImageLayout::DEPTH_STENCIL_READ_ONLY)
			->build()
		);
		registerPass(Pass::SSAO, RenderPass::Builder::get()
			->withColourAttachment(Texture::SSAO, LoadOp::CLEAR, StoreOp::STORE, ImageLayout::SHADER_READ_ONLY)
			->usesDescriptorInShader(ImageType::COLOR | ImageType::DEPTH) // Uses depth and normals G-buffer
			->build()
		);
		registerPass(Pass::FORWARD, RenderPass::Builder::get()
			->withColourAttachment(Texture::MAIN_HDR, LoadOp::LOAD, StoreOp::STORE, ImageLayout::COLOR, ImageLayout::COLOR)
			->withColourAttachment(Texture::BRIGHTNESS, LoadOp::CLEAR, StoreOp::STORE, ImageLayout::COLOR)
			->withDepthAttachment(Texture::DEPTH, LoadOp::CLEAR, StoreOp::DONT_CARE, ImageLayout::DEPTH)
			->usesDescriptorInShader(ImageType::COLOR) // Uses SSAO texture in shader
			->build()
		);
		registerPass(Pass::DEFERRED_WRITE, RenderPass::Builder::get()
			->withColourAttachment(Texture::GBUFFER1, LoadOp::CLEAR, StoreOp::STORE, ImageLayout::SHADER_READ_ONLY)
			->withColourAttachment(Texture::GBUFFER2, LoadOp::CLEAR, StoreOp::STORE, ImageLayout::SHADER_READ_ONLY)
			->withColourAttachment(Texture::GBUFFER3, LoadOp::CLEAR, StoreOp::STORE, ImageLayout::SHADER_READ_ONLY)
			->withDepthAttachment(Texture::DEPTH, LoadOp::CLEAR, StoreOp::STORE, ImageLayout::DEPTH_STENCIL_READ_ONLY)
			->build()
		);
		registerPass(Pass::DEFERRED_SHADE, RenderPass::Builder::get()
			->withColourAttachment(Texture::MAIN_HDR, LoadOp::LOAD, StoreOp::STORE, ImageLayout::COLOR)
			->withColourAttachment(Texture::BRIGHTNESS, LoadOp::CLEAR, StoreOp::STORE, ImageLayout::COLOR)
			->usesDescriptorInShader(ImageType::COLOR) // Uses SSAO texture in shader
			->build()
		);
		registerPass(Pass::POST_PROCESS_HDR, RenderPass::Builder::get()
			->withColourAttachment(Texture::MAIN_SDR, LoadOp::CLEAR, StoreOp::STORE, ImageLayout::COLOR)
			->usesDescriptorInShader(ImageType::COLOR) // Uses a HDR texture in shader
			->build()
		);
		registerPass(Pass::POST_PROCESS_LDR, RenderPass::Builder::get()
			->withColourAttachment(Texture::MAIN_SDR, LoadOp::CLEAR, StoreOp::STORE, ImageLayout::COLOR)
			->usesDescriptorInShader(ImageType::COLOR) // Uses an SDR texture in shader
			->build()
		);
		registerPass(Pass::DEBUG, RenderPass::Builder::get()
			->withColourAttachment(Texture::MAIN_HDR, LoadOp::CLEAR, StoreOp::STORE, ImageLayout::COLOR)
			->withDepthAttachment(Texture::DEPTH, LoadOp::CLEAR, StoreOp::DONT_CARE, ImageLayout::DEPTH_STENCIL_READ_ONLY)
			->build()
		);
		registerPass(Pass::DEBUG_SHAPES, RenderPass::Builder::get()
			->withColourAttachment(Texture::MAIN_HDR, LoadOp::LOAD, StoreOp::STORE, ImageLayout::COLOR)
			->withDepthAttachment(Texture::DEPTH, LoadOp::LOAD, StoreOp::DONT_CARE, ImageLayout::DEPTH_STENCIL_READ_ONLY)
			->build()
		);
	}

	void initialiseDeferredPasses(const Swapchain& swapchain) {
		// GUI pass needs to know what format the swapchain is in
		// since it writes directly to the swapchain image
		registerPass(Pass::GUI, RenderPass::Builder::get()
			->withColourAttachment(Texture::SWAPCHAIN, LoadOp::LOAD, StoreOp::STORE, ImageLayout::PRESENT)
			->usesDescriptorInShader(ImageType::COLOR) // Uses swapchain image in shader
			->build()
		);
	}

	void registerPass(Pass key, RenderPass renderPass) {
		renderPasses[key] = std::move(renderPass);
	}

	RenderPass& get(Pass pass) {
		try {
			return renderPasses.at(pass);
		} catch (const std::out_of_range& e) {
			throw Utils::Error("RenderPasses: Could not find render pass '%d'! Maybe it has not been initialised?\n", pass);
		}
	}

}