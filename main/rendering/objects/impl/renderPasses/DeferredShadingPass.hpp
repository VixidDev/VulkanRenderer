#pragma once

#include "../../base/RenderPass.hpp"

class DeferredShadingPass : public RenderPass {
public:
	DeferredShadingPass(VulkanWindow* window, VkSampleCountFlagBits* sampleCount);

	void recreate();
private:
};