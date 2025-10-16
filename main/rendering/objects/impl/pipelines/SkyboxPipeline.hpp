#pragma once

#include "../../base/Pipeline.hpp"

class SkyboxPipeline : public Pipeline {
public:
	SkyboxPipeline(
		VulkanWindow* window,
		PipelineLayout* pipelineLayout,
		RenderPass* renderPass);

	void recreate();
private:
};