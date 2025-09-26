#pragma once

#include "../../base/Pipeline.hpp"

class MosaicPipeline : public Pipeline {
public:
	MosaicPipeline(
		VulkanWindow* window,
		PipelineLayout* pipelineLayout,
		RenderPass* renderPass);

	void recreate();
private:
};