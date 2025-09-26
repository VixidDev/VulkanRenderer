#pragma once

#include "../../base/Pipeline.hpp"

class LineDebugPipeline : public Pipeline {
public:
	LineDebugPipeline(
		VulkanWindow* window,
		PipelineLayout* pipelineLayout,
		RenderPass* renderPass,
		VkSampleCountFlagBits* sampleCount);

	void recreate();
private:
	bool* shadowsEnabled = nullptr;
};