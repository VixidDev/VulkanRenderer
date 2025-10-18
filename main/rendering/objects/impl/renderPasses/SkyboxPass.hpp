#pragma once

#include "../../base/RenderPass.hpp"

class SkyboxPass : public RenderPass {
public:
	SkyboxPass(VulkanWindow* window);

	void recreate();
private:
};