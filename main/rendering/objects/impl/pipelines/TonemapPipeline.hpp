#pragma once

#include "../../base/Pipeline.hpp"

class TonemapPipeline : public Pipeline {
public:
	TonemapPipeline(VulkanWindow* window, PipelineLayout* pipelineLayout, RenderPass* renderPass);

	void recreate();
private:
};