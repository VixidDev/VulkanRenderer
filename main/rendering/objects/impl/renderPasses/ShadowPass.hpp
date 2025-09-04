#pragma once

#include "../../base/RenderPass.hpp"

class ShadowPass : public RenderPass {
public:
	ShadowPass(VulkanWindow* window, VkSampleCountFlagBits* sampleCount);

	void recreate();
private:
};