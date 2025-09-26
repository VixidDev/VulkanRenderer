#pragma once

#include "../../base/Pipeline.hpp"

class DebugViewsPipeline : public Pipeline {
public:
	DebugViewsPipeline(
		VulkanWindow* window,
		PipelineLayout* pipelineLayout,
		RenderPass* renderPass,
		VkSampleCountFlagBits* sampleCount);

	void recreate();
private:
};