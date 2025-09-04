#pragma once

#include "objects/base/RenderPass.hpp"
#include "objects/base/Framebuffer.hpp"

namespace RendererUtils {

	void beginRenderPass(VkCommandBuffer cmdBuff, RenderPass* renderPass, Framebuffer* framebuffer, std::uint32_t imageIndex);
	void endRenderPass(VkCommandBuffer cmdBuff);

}