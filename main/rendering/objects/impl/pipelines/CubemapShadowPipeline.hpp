#pragma once

#include "../../base/Pipeline.hpp"

using _PipelineLayout = std::unique_ptr<PipelineLayout>;
using _RenderPass = std::unique_ptr<RenderPass>;

class CubemapShadowPipeline : public Pipeline {
public:
	CubemapShadowPipeline(
		VulkanWindow* window,
		_PipelineLayout* pipelineLayout,
		_RenderPass* renderPass,
		VkSampleCountFlagBits* sampleCount,
		VkExtent2D* shadowMapResolution);

	void recreate();
private:
};