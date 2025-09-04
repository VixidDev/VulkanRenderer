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

	std::unique_ptr<PipelineLayout>* pipelineLayout = nullptr;
	std::unique_ptr<RenderPass>* renderPass = nullptr;

	VkExtent2D* renderExtent = nullptr;
	VkSampleCountFlagBits* sampleCount = nullptr;
};