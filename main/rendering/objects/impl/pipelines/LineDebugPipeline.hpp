#pragma once

#include "../../base/Pipeline.hpp"

class RenderPass;

using _PipelineLayout = std::unique_ptr<PipelineLayout>;
using _RenderPass = std::unique_ptr<RenderPass>;

class LineDebugPipeline : public Pipeline {
public:
	LineDebugPipeline(
		VulkanWindow* window,
		_PipelineLayout* pipelineLayout,
		_RenderPass* renderPass,
		VkSampleCountFlagBits* sampleCount);

	void recreate();
private:
	bool* shadowsEnabled = nullptr;
};