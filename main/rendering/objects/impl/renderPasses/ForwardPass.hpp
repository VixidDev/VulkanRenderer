#pragma once

#include "../../base/RenderPass.hpp"

class ForwardPass : public RenderPass {
public:
	ForwardPass(VulkanWindow* window);

	void recreate();
private:
};