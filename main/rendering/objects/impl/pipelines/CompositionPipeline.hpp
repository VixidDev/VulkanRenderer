#pragma once

#include "../../base/Pipeline.hpp"

class CompositionPipeline : public Pipeline {
public:
	CompositionPipeline(
		VulkanWindow* window,
		PipelineLayout* pipelineLayout,
		RenderPass* renderPass);

	void recreate();
private:
};