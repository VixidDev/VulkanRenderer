#pragma once

#include <vector>

#include "../../../vulkan/objects/VkObjects.hpp"

class VulkanWindow;

class Framebuffer {
public:
	Framebuffer() = default;
	Framebuffer(VulkanWindow* window);

	virtual ~Framebuffer() = default;

	virtual void recreate();

	VkFramebuffer getHandle(std::uint32_t imageIndex);
	VkExtent2D getRenderExtent();
protected:
	VulkanWindow* window;

	std::vector<vk::Framebuffer> framebuffers;

	VkExtent2D* renderExtent = nullptr;
};