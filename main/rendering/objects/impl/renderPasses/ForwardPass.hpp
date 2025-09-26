#pragma once

#include "../../base/RenderPass.hpp"

class ForwardPass : public RenderPass {
public:
	ForwardPass(VulkanWindow* window, VkSampleCountFlagBits* sampleCount);

	void recreate();
private:
	void recreateNonMSAA();
	void recreateMSAA();
};