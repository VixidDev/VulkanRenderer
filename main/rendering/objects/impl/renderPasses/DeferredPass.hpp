#pragma once

#include "../../base/RenderPass.hpp"

class DeferredPass : public RenderPass {
public:
	DeferredPass(VulkanWindow* window, VkSampleCountFlagBits* sampleCount);

	void recreate();
private:
};