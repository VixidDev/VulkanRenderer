#pragma once

#include "../../base/Pipeline.hpp"

class DebugViewsPipeline : public Pipeline {
public:
	DebugViewsPipeline(
		VulkanWindow* window,
		PipelineLayout* pipelineLayout,
		RenderPass* renderPass);

	void recreate();
private:
};