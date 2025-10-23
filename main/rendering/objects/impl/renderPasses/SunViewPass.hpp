#pragma once

#include "../../base/RenderPass.hpp"

class SunViewPass : public RenderPass {
public:
	SunViewPass(VulkanWindow* window);

	void recreate();
private:
};