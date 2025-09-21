#pragma once

#include "objects/base/RenderPass.hpp"
#include "objects/base/Framebuffer.hpp"

namespace RendererUtils {

	void bindCommandBuffer(VkCommandBuffer cmdBuff);

	void beginRenderPass(RenderPass* renderPass, Framebuffer* framebuffer, std::uint32_t imageIndex);

	void bindGraphicPipeline(VkPipeline pipeline);
	void bindGraphicDescriptorSets(
		VkPipelineLayout pipelineLayout, 
		std::uint32_t firstSet, 
		std::uint32_t descSetCount,
		const VkDescriptorSet* pDescriptorSets, 
		std::uint32_t dynamicOffsetCount = 0, 
		const std::uint32_t* pDynamicOffsets = nullptr);
	void bindPushConstant(
		VkPipelineLayout pipelineLayout,
		VkShaderStageFlags stageFlags,
		std::uint32_t offset,
		std::uint32_t size,
		const void* pValues);

	void endRenderPass();

	void checkCommandBuffer();

}