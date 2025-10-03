#pragma once

#include <map>
#include <string>
#include <memory>

#include "../../base/Framebuffer.hpp"
#include "../../base/TextureBuffer.hpp"
#include "../../base/RenderPass.hpp"

class CubemapArrayFramebuffer : public Framebuffer {
public:
	CubemapArrayFramebuffer(
		VulkanWindow* window,
		TextureBuffer* textureBuffer,
		RenderPass* renderPass,
		std::uint32_t arraySize,
		VkExtent2D* renderExtent);

	void recreate();
private:
	TextureBuffer* textureBuffer = nullptr;
	RenderPass* renderPass = nullptr;

	std::uint32_t arraySize = 1;
};