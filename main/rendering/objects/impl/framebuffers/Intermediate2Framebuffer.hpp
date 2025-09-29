#pragma once

#include <map>
#include <string>
#include <memory>

#include "../../base/Framebuffer.hpp"
#include "../../base/TextureBuffer.hpp"
#include "../../base/RenderPass.hpp"

using _TextureBuffer = std::unique_ptr<TextureBuffer>;

class Intermediate2Framebuffer : public Framebuffer {
public:
	Intermediate2Framebuffer(VulkanWindow* window, std::map<std::string, _TextureBuffer>* textureBuffers, RenderPass* renderPass);

	void recreate();
private:
	std::map<std::string, _TextureBuffer>* textureBuffers = nullptr;
	RenderPass* renderPass = nullptr;
};