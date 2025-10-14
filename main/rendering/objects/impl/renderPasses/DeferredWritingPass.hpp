#pragma once

#include "../../base/RenderPass.hpp"

class DeferredWritingPass : public RenderPass {
public:
	DeferredWritingPass(VulkanWindow* window, VkSampleCountFlagBits* sampleCount);

	void recreate();
private:
};