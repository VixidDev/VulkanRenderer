#pragma once

#include "../../base/Pipeline.hpp"

class SSAOBlurPipeline : public Pipeline {
public:
	SSAOBlurPipeline(VulkanWindow* window, PipelineLayout* pipelineLayout, RenderPass* renderPass);

	void recreate();
private:
};