#include "RendererUtils.hpp"

namespace RendererUtils {

	VkCommandBuffer boundCommandBuffer = VK_NULL_HANDLE;

	void bindCommandBuffer(VkCommandBuffer cmdBuff) {
		boundCommandBuffer = cmdBuff;
	}

	void beginRenderPass(RenderPass* renderPass, Framebuffer* framebuffer, std::uint32_t imageIndex) {
		checkCommandBuffer();

		VkRenderPassBeginInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		passInfo.renderPass = renderPass->getRenderPassHandle();
		passInfo.framebuffer = framebuffer->getHandle(imageIndex);
		passInfo.renderArea.offset = VkOffset2D{ 0, 0 };
		passInfo.renderArea.extent = framebuffer->getRenderExtent();
		passInfo.clearValueCount = static_cast<std::uint32_t>(renderPass->getClearValues().size());
		passInfo.pClearValues = renderPass->getClearValues().data();

		vkCmdBeginRenderPass(boundCommandBuffer, &passInfo, VK_SUBPASS_CONTENTS_INLINE);
	}

	void bindGraphicPipeline(VkPipeline pipeline) {
		checkCommandBuffer();

		vkCmdBindPipeline(boundCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
	}

	void bindGraphicDescriptorSets(
		VkPipelineLayout pipelineLayout, 
		std::uint32_t firstSet, 
		std::uint32_t descSetCount, 
		const VkDescriptorSet* pDescriptorSets, 
		std::uint32_t dynamicOffsetCount, /* default = 0 */
		const std::uint32_t* pDynamicOffsets /* default = nullptr */)
	{
		checkCommandBuffer();

		vkCmdBindDescriptorSets(
			boundCommandBuffer, 
			VK_PIPELINE_BIND_POINT_GRAPHICS, 
			pipelineLayout, 
			firstSet, 
			descSetCount, 
			pDescriptorSets, 
			dynamicOffsetCount, 
			pDynamicOffsets
		);
	}

	void bindPushConstant(
		VkPipelineLayout pipelineLayout, 
		VkShaderStageFlags stageFlags, 
		std::uint32_t offset, 
		std::uint32_t size, 
		const void* pValues) 
	{
		checkCommandBuffer();

		vkCmdPushConstants(boundCommandBuffer, pipelineLayout, stageFlags, offset, size, pValues);
	}

	void endRenderPass() {
		vkCmdEndRenderPass(boundCommandBuffer);
	}

	void checkCommandBuffer() {
		if (boundCommandBuffer == VK_NULL_HANDLE) {
			std::fprintf(stderr, "Running Vulkan command without a bound command buffer!\n");
		}
	}

}