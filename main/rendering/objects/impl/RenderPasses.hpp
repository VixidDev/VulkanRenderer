#pragma once

#include "../base/RenderPass.hpp"

enum Pass {
	FORWARD,
	DEFERRED_WRITE,
	DEFERRED_SHADE,
};

namespace RenderPasses {

	void initialise();

	void registerPass(Pass key, RenderPass::Builder* pendingRenderPass);


}