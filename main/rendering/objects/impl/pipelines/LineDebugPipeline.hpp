#pragma once

#include "../../base/Pipeline.hpp"

class LineDebugPipeline : public Pipeline {
public:
	LineDebugPipeline(
		VulkanWindow* window,
		PipelineLayout* pipelineLayout,
		RenderPass* renderPass);

	void recreate();
private:
	bool* shadowsEnabled = nullptr;
};