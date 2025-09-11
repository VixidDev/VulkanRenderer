#pragma once

#include <map>
#include <string>
#include <memory>

#include "../../base/Framebuffer.hpp"
#include "../../base/TextureBuffer.hpp"
#include "../../base/RenderPass.hpp"

using _TextureBuffer = std::unique_ptr<TextureBuffer>;
using _RenderPass = std::unique_ptr<RenderPass>;

class CubemapArrayFramebuffer : public Framebuffer {
public:
	CubemapArrayFramebuffer(
		VulkanWindow* window,
		_TextureBuffer* textureBuffer,
		_RenderPass* renderPass,
		std::uint32_t arraySize,
		VkExtent2D* shadowMapResolution);

	void recreate();
private:
	_TextureBuffer* textureBuffer = nullptr;
	_RenderPass* renderPass = nullptr;

	std::uint32_t arraySize = 1;
};