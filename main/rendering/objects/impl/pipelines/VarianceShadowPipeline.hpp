#pragma once

#include "../../base/Pipeline.hpp"

class VarianceShadowPipeline : public Pipeline {
public:
	VarianceShadowPipeline(
		VulkanWindow* window,
		PipelineLayout* pipelineLayout,
		RenderPass* renderPass,
		VkExtent2D* shadowMapResolution,
		int lightType);

	void recreate();
private:
	int lightType;
};