#pragma once

#include "../../base/Pipeline.hpp"

class SunPipeline : public Pipeline {
public:
	SunPipeline(VulkanWindow* window, PipelineLayout* pipelineLayout, RenderPass* renderPass);

	void recreate();
private:
};