#pragma once

#include "../../base/Pipeline.hpp"

class DeferredShadingPipeline : public Pipeline {
public:
	DeferredShadingPipeline(
		VulkanWindow* window,
		PipelineLayout* pipelineLayout,
		RenderPass* renderPass,
		bool* shadowsEnabled,
		bool* useViewSpaceNormals);

	void recreate();
private:
	bool* shadowsEnabled = nullptr;
	bool* useViewSpaceNormals = nullptr;
	int viewSpaceNormals = 0;
};