#pragma once

#include "../../base/RenderPass.hpp"

class VarianceShadowBlurPass : public RenderPass {
public:
	VarianceShadowBlurPass(VulkanWindow* window);

	void recreate();
private:
};