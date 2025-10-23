#pragma once

#include "PipelineLayout.hpp"
#include "RenderPass.hpp"
#include "../../../vulkan/objects/VkObjects.hpp"

#include <memory>

class Pipeline {
public:
	Pipeline() = default;
	Pipeline(VulkanWindow* window);

	virtual ~Pipeline() = default;

	virtual void recreate();

	VkPipeline getHandle();
protected:
	VulkanWindow* window;

	vk::Pipeline pipeline;

	PipelineLayout* pipelineLayout = nullptr;
	RenderPass* renderPass = nullptr;

	VkExtent2D* renderExtent = nullptr;
};