#pragma once

#include "../../base/RenderPass.hpp"

class DebugPass : public RenderPass {
public:
	DebugPass(VulkanWindow* window);

	void recreate();
private:
};