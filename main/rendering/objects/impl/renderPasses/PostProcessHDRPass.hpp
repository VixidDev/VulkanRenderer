#pragma once

#include "../../base/RenderPass.hpp"

class PostProcessHDRPass : public RenderPass {
public:
	PostProcessHDRPass(VulkanWindow* window);

	void recreate();
private:
};