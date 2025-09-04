#pragma once

#include "../../base/RenderPass.hpp"

class GUIPass : public RenderPass {
public:
	GUIPass(VulkanWindow* window, VkSampleCountFlagBits* sampleCount);

	void recreate();
private:
};