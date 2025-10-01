#pragma once

#include "../../base/Pipeline.hpp"

class SunViewPipeline : public Pipeline {
public:
	SunViewPipeline(
		VulkanWindow* window,
		PipelineLayout* pipelineLayout,
		RenderPass* renderPass,
		VkSampleCountFlagBits* sampleCount);

	void recreate();
private:
};