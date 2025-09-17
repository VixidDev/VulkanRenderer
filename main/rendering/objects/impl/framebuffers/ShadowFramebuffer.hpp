#pragma once

#include <map>
#include <string>
#include <memory>

#include "../../base/Framebuffer.hpp"
#include "../../base/TextureBuffer.hpp"
#include "../../base/RenderPass.hpp"

using _TextureBuffer = std::unique_ptr<TextureBuffer>;

class ShadowFramebuffer : public Framebuffer {
public:
	ShadowFramebuffer(
	VulkanWindow* window,
	std::map<std::string, _TextureBuffer>* textureBuffers,
	RenderPass* renderPass,
	VkExtent2D* shadowMapResolution);

	void recreate();
private:
	std::map<std::string, _TextureBuffer>* textureBuffers = nullptr;
	RenderPass* renderPass = nullptr;
};