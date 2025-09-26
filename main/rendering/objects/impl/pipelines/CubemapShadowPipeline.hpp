#pragma once

#include "../../base/Pipeline.hpp"

class CubemapShadowPipeline : public Pipeline {
public:
	CubemapShadowPipeline(
		VulkanWindow* window,
		PipelineLayout* pipelineLayout,
		RenderPass* renderPass,
		VkSampleCountFlagBits* sampleCount,
		VkExtent2D* shadowMapResolution);

	void recreate();
private:
};