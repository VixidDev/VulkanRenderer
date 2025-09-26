#pragma once

#include <vector>

#include "../../../vulkan/objects/VkObjects.hpp"

class VulkanWindow;

class RenderPass {
public:
	RenderPass() = default;
	RenderPass(VulkanWindow* window);

	virtual ~RenderPass() = default;

	virtual void recreate();

	vk::RenderPass& getRenderPass();
	VkRenderPass getRenderPassHandle();
	std::vector<VkClearValue>& getClearValues();
protected:
	VulkanWindow* window;

	vk::RenderPass renderPass;
	std::vector<VkClearValue> clearValues;

	// Pointer to current sample count setting
	VkSampleCountFlagBits* sampleCount = nullptr;
};