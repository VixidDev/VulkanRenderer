#pragma once

#include <map>
#include <string>

#include "../../base/Framebuffer.hpp"
#include "../../base/TextureBuffer.hpp"
#include "../../base/RenderPass.hpp"

class ArrayFramebuffer : public Framebuffer {
public:
	ArrayFramebuffer(
		VulkanWindow* window,
		std::initializer_list<TextureBuffer*> textureBuffers,
		RenderPass* renderPass,
		std::uint32_t arraySize,
		VkExtent2D* renderExtent);

	void recreate();
private:
	std::vector<TextureBuffer*> textureBuffers;
	RenderPass* renderPass = nullptr;

	std::uint32_t arraySize = 1;
};