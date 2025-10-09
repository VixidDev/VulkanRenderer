#pragma once

#include "../../base/RenderPass.hpp"

class PreSSAOPass : public RenderPass {
public:
	PreSSAOPass(VulkanWindow* window);

	void recreate();
private:
};