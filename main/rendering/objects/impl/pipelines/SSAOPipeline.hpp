#pragma once

#include "../../base/Pipeline.hpp"

class SSAOPipeline : public Pipeline {
public:
	SSAOPipeline(VulkanWindow* window, PipelineLayout* pipelineLayout, RenderPass* renderPass);

	void recreate();
private:
};