#pragma once

#include "../base/RenderPass.hpp"

enum Pass {
	SHADOW,
	SHADOW_VSM,
	SHADOW_VSM_BLUR,
	SKYBOX,
	SUN,
	PRE_SSAO,
	SSAO,
	FORWARD,
	DEFERRED_WRITE,
	DEFERRED_SHADE,
	GUI,

	POST_PROCESS_HDR,
	POST_PROCESS_LDR,

	DEBUG,
	DEBUG_SHAPES
};

namespace RenderPasses {

	void initialise();
	void registerPass(Pass key, RenderPass pendingRenderPass);

}