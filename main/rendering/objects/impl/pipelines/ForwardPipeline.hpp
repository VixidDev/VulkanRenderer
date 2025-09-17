#pragma once

#include "../../base/Pipeline.hpp"

class ForwardPipeline : public Pipeline {
public:
	ForwardPipeline(
		VulkanWindow* window,
		PipelineLayout* pipelineLayout,
		RenderPass* renderPass,
		VkSampleCountFlagBits* sampleCount,
		bool* shadowsEnabled);

	void recreate();
private:
	bool* shadowsEnabled = nullptr;
};