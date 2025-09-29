#pragma once

#include "../../base/Pipeline.hpp"

class BloomPipeline : public Pipeline {
public:
	BloomPipeline(
		VulkanWindow* window,
		PipelineLayout* pipelineLayout,
		RenderPass* renderPass);

	void recreate();
private:
};