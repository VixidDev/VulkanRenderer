#pragma once

#include "../../base/Framebuffer.hpp"
#include "../../base/TextureBuffer.hpp"
#include "../../base/RenderPass.hpp"

class WriteToTargetFramebuffer : public Framebuffer {
public:
	WriteToTargetFramebuffer(VulkanWindow* window, TextureBuffer* textureBuffer, RenderPass* renderPass, VkExtent2D* renderExtent = nullptr);

	void recreate();

	TextureBuffer* getRenderTarget();
private:
	TextureBuffer* textureBuffer = nullptr;
	RenderPass* renderPass = nullptr;
};