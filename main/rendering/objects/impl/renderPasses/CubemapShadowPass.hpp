#pragma once

#include "../../base/RenderPass.hpp"

class CubemapShadowPass : public RenderPass {
public:
	CubemapShadowPass(VulkanWindow* window, VkSampleCountFlagBits* sampleCount);

	void recreate();
private:
};