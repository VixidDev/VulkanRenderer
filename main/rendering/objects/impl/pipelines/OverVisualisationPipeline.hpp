#pragma once

#include "../../base/Pipeline.hpp"

class OverVisualisationPipeline : public Pipeline {
public:
	OverVisualisationPipeline(
		VulkanWindow* window,
		PipelineLayout* pipelineLayout,
		RenderPass* renderPass,
		VkSampleCountFlagBits* sampleCount);

	void recreate();
private:
};