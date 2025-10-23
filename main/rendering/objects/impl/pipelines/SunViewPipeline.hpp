#pragma once

#include "../../base/Pipeline.hpp"

class SunViewPipeline : public Pipeline {
public:
	SunViewPipeline(
		VulkanWindow* window,
		PipelineLayout* pipelineLayout,
		RenderPass* renderPass);

	void recreate();
private:
};