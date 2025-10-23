#pragma once

#include "../../base/Pipeline.hpp"

class ForwardPipeline : public Pipeline {
public:
	ForwardPipeline(
		VulkanWindow* window,
		PipelineLayout* pipelineLayout,
		RenderPass* renderPass,
		bool* shadowsEnabled,
		int* vsmShadowsEnabled);

	void recreate();
private:
	bool* shadowsEnabled = nullptr;
	int* vsmShadowsEnabled = nullptr;
};