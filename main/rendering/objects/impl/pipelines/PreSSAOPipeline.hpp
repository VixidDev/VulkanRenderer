#pragma once

#include "../../base/Pipeline.hpp"

class PreSSAOPipeline : public Pipeline {
public:
	PreSSAOPipeline(VulkanWindow* window, PipelineLayout* pipelineLayout, RenderPass* renderPass);

	void recreate();
private:
};