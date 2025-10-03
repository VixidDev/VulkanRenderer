#pragma once

#include <map>
#include <string>
#include <memory>

#include "../../base/Framebuffer.hpp"
#include "../../base/TextureBuffer.hpp"
#include "../../base/RenderPass.hpp"

class CubemapFramebuffer : public Framebuffer {
public:
	CubemapFramebuffer(
		VulkanWindow* window,
		TextureBuffer* textureBuffer,
		RenderPass* renderPass,
		VkExtent2D* renderExtent);

	void recreate();
private:
	TextureBuffer* textureBuffer = nullptr;
	RenderPass* renderPass = nullptr;
};