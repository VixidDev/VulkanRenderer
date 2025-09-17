#pragma once

#include <memory>

#include "../../base/Framebuffer.hpp"
#include "../../base/RenderPass.hpp"

class GUIFramebuffer : public Framebuffer {
public:
	GUIFramebuffer(VulkanWindow* window, RenderPass* renderPass);

	void recreate();

private:
	RenderPass* renderPass = nullptr;
};