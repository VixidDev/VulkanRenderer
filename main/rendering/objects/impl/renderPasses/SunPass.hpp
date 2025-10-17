#pragma once

#include "../../base/RenderPass.hpp"

class SunPass : public RenderPass {
public:
	SunPass(VulkanWindow* window);

	void recreate();
private:
};