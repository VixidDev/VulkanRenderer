#pragma once

#include "../../base/RenderPass.hpp"

class VarianceShadowPass : public RenderPass {
public:
	VarianceShadowPass(VulkanWindow* window);

	void recreate();
private:
};