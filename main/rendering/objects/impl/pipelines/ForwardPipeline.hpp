#pragma once

#include "../../base/Pipeline.hpp"

class ForwardPipeline : public Pipeline {
public:
	ForwardPipeline(
		VulkanWindow* window,
		PipelineLayout* pipelineLayout,
		RenderPass* renderPass,
		bool* shadowsEnabled,
		int* vsmShadowsEnabled,
		int* numLights);

	void recreate();
private:
	bool* shadowsEnabled = nullptr;
	int* vsmShadowsEnabled = nullptr;
	int* numLights = nullptr;
};