#pragma once

#include "../../base/Pipeline.hpp"

class VarianceShadowPipeline : public Pipeline {
public:
	VarianceShadowPipeline(
		VulkanWindow* window,
		PipelineLayout* pipelineLayout,
		RenderPass* renderPass,
		VkSampleCountFlagBits* sampleCount,
		VkExtent2D* shadowMapResolution);

	void recreate();
private:
};