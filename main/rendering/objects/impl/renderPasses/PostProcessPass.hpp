#pragma once

#include "../../base/RenderPass.hpp"

class PostProcessPass : public RenderPass {
public:
	PostProcessPass(VulkanWindow* window);

	void recreate();
private:
};