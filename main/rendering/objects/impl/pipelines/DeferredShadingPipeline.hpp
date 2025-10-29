#pragma once

#include "../../base/Pipeline.hpp"

class DeferredShadingPipeline : public Pipeline {
public:
	DeferredShadingPipeline(
		VulkanWindow* window,
		PipelineLayout* pipelineLayout,
		RenderPass* renderPass,
		bool* shadowsEnabled,
		int* vsmShadowsEnabled,
		bool* useViewSpaceNormals,
		int* numLights);

	void recreate();
private:
	bool* shadowsEnabled = nullptr;
	int* vsmShadowsEnabled = nullptr;
	bool* useViewSpaceNormals = nullptr;
	int viewSpaceNormals = 0;
	int* numLights = nullptr;
};