#pragma once

#include "../../base/Pipeline.hpp"

class FXAAPipeline : public Pipeline {
public:
	FXAAPipeline(VulkanWindow* window, PipelineLayout* pipelineLayout, RenderPass* renderPass);

	void recreate();
private:
};