#pragma once

#include "../../../vulkan/objects/VkObjects.hpp"

class VulkanWindow;

class PipelineLayout {
public:
	PipelineLayout() = default;
	PipelineLayout(VulkanWindow* window);

	virtual ~PipelineLayout() = default;

	virtual void recreate();

	VkPipelineLayout getHandle();
protected:
	VulkanWindow* window;

	vk::PipelineLayout pipelineLayout;
};
