#pragma once

#include "../../base/RenderPass.hpp"

class PostProcessLDRPass : public RenderPass {
public:
	PostProcessLDRPass(VulkanWindow* window);

	void recreate();
private:
};