#pragma once

#include "../../base/Pipeline.hpp"

class BlurPipeline : public Pipeline {
public:
	BlurPipeline(
		VulkanWindow* window,
		PipelineLayout* pipelineLayout,
		RenderPass* renderPass,
		int* kernelSize,
		VkExtent2D* renderExtent = nullptr);

	void recreate();
private:
	int* kernelSizePtr = nullptr;
	int kernelSize = 0;
};