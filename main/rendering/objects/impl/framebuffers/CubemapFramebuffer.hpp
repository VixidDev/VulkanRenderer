#pragma once

#include <map>
#include <string>
#include <memory>

#include "../../base/Framebuffer.hpp"
#include "../../base/TextureBuffer.hpp"
#include "../../base/RenderPass.hpp"

using _TextureBuffer = std::unique_ptr<TextureBuffer>;
using _RenderPass = std::unique_ptr<RenderPass>;

class CubemapFramebuffer : public Framebuffer {
public:
	CubemapFramebuffer(
		VulkanWindow* window,
		std::map<std::string, _TextureBuffer>* textureBuffers,
		_RenderPass* renderPass,
		VkExtent2D* shadowMapResolution);

	void recreate();
private:
	std::map<std::string, _TextureBuffer>* textureBuffers = nullptr;
	_RenderPass* renderPass = nullptr;
};