#pragma once

#include "../../base/Pipeline.hpp"

class DeferredShadingPipeline : public Pipeline {
public:
	DeferredShadingPipeline(
		VulkanWindow* window,
		PipelineLayout* pipelineLayout,
		RenderPass* renderPass,
		VkSampleCountFlagBits* sampleCount,
		bool* shadowsEnabled);

	void recreate();
private:
	bool* shadowsEnabled = nullptr;
};