#pragma once

#include "../../base/RenderPass.hpp"

class DeferredShadingPass : public RenderPass {
public:
	DeferredShadingPass(VulkanWindow* window);

	void recreate();
private:
};