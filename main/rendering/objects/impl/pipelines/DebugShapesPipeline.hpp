#pragma once

#include "../../base/Pipeline.hpp"

class DebugShapesPipeline : public Pipeline {
public:
	DebugShapesPipeline(VulkanWindow* window, PipelineLayout* pipelineLayout, RenderPass* renderPass);

	void recreate();
};