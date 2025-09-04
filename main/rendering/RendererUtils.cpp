#include "RendererUtils.hpp"

namespace RendererUtils {

	void beginRenderPass(VkCommandBuffer cmdBuff, RenderPass* renderPass, Framebuffer* framebuffer, std::uint32_t imageIndex) {
		VkRenderPassBeginInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		passInfo.renderPass = renderPass->getRenderPassHandle();
		passInfo.framebuffer = framebuffer->getHandle(imageIndex);
		passInfo.renderArea.offset = VkOffset2D{ 0, 0 };
		passInfo.renderArea.extent = framebuffer->getRenderExtent();
		passInfo.clearValueCount = static_cast<std::uint32_t>(renderPass->getClearValues().size());
		passInfo.pClearValues = renderPass->getClearValues().data();

		vkCmdBeginRenderPass(cmdBuff, &passInfo, VK_SUBPASS_CONTENTS_INLINE);
	}

	void endRenderPass(VkCommandBuffer cmdBuff) {
		vkCmdEndRenderPass(cmdBuff);
	}

}