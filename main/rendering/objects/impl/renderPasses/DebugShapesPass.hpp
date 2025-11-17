#pragma once

#include "../../base/RenderPass.hpp"

class DebugShapesPass : public RenderPass {
public:
	DebugShapesPass(VulkanWindow* window);

	void recreate();
private:
};