#pragma once

#include <memory>

#include "../../base/Framebuffer.hpp"
#include "../../base/RenderPass.hpp"

using _RenderPass = std::unique_ptr<RenderPass>;

class GUIFramebuffer : public Framebuffer {
public:
	GUIFramebuffer(VulkanWindow* window, _RenderPass* renderPass);

	void recreate();

private:
	_RenderPass* renderPass = nullptr;
};