#pragma once

#include <map>
#include <string>
#include <memory>

#include "../../base/Framebuffer.hpp"
#include "../../base/TextureBuffer.hpp"
#include "../../base/RenderPass.hpp"

using _TextureBuffer = std::unique_ptr<TextureBuffer>;

class SunFramebuffer : public Framebuffer {
public:
	SunFramebuffer(
		VulkanWindow* window,
		std::map<std::string, _TextureBuffer>* textureBuffers,
		RenderPass* renderPass,
		VkSampleCountFlagBits* sampleCount);

	void recreate();

private:
	std::map<std::string, _TextureBuffer>* textureBuffers = nullptr;
	RenderPass* renderPass = nullptr;
	VkSampleCountFlagBits* sampleCount = nullptr;
};