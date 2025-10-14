#pragma once

#include "../../base/Pipeline.hpp"

class DeferredWritingPipeline : public Pipeline {
public:
	DeferredWritingPipeline(
		VulkanWindow* window,
		PipelineLayout* pipelineLayout,
		RenderPass* renderPass,
		VkSampleCountFlagBits* sampleCount,
		bool* useViewSpaceNormals = nullptr);

	void recreate();
private:
	bool* useViewSpaceNormals = nullptr;
	int viewSpaceNormals = 0;
};