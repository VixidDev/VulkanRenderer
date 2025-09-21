#pragma once

#include <map>
#include <string>

#include "../../base/Framebuffer.hpp"
#include "../../base/TextureBuffer.hpp"
#include "../../base/RenderPass.hpp"

class ShadowFramebuffer : public Framebuffer {
public:
	ShadowFramebuffer(
	VulkanWindow* window,
	std::initializer_list<TextureBuffer*> textureBuffers,
	RenderPass* renderPass,
	VkExtent2D* shadowMapResolution);

	void recreate();
private:
	std::vector<TextureBuffer*> textureBuffers;
	RenderPass* renderPass = nullptr;
};