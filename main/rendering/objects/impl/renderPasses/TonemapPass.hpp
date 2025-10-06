#pragma once

#include "../../base/RenderPass.hpp"

class TonemapPass : public RenderPass {
public:
	TonemapPass(VulkanWindow* window);

	void recreate();
private:
};