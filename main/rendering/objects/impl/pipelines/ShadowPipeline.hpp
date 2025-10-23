#pragma once

#include "../../base/Pipeline.hpp"

class ShadowPipeline : public Pipeline {
public:
	ShadowPipeline(
		VulkanWindow* window,
		PipelineLayout* pipelineLayout,
		RenderPass* renderPass,
		VkExtent2D* shadowMapResolution);

	void recreate();
private:
};