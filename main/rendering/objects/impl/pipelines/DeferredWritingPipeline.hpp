#pragma once

#include "../../base/Pipeline.hpp"

class DeferredWritingPipeline : public Pipeline {
public:
	DeferredWritingPipeline(
		VulkanWindow* window,
		PipelineLayout* pipelineLayout,
		RenderPass* renderPass,
		VkSampleCountFlagBits* sampleCount);

	void recreate();
private:
};