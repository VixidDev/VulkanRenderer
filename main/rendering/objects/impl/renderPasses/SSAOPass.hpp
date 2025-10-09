#pragma once

#include "../../base/RenderPass.hpp"

class SSAOPass : public RenderPass {
public:
	SSAOPass(VulkanWindow* window);

	void recreate();
private:
};