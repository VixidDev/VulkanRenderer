#pragma once

#include "../../base/RenderPass.hpp"

class DeferredWritingPass : public RenderPass {
public:
	DeferredWritingPass(VulkanWindow* window);

	void recreate();
private:
};