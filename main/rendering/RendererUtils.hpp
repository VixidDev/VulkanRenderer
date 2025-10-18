#pragma once

#include "Renderer.hpp"

namespace RendererUtils {

	void checkCommandBuffer();

	// Command buffer
	void bindCommandBuffer(std::vector<VkCommandBuffer>& cmdBuffs, std::uint32_t frameIndex);
	void beginCommandBuffer(VkCommandBufferUsageFlags usageFlags = 0);
	void endCommandBuffer();

	// Buffers
	void updateUniformBuffer(IUniformBuffer* uniformBuffer);
	void updateShaderStorageBuffer(IShaderStorageBuffer* shaderStorageBuffer);

	// Render passes
	void beginRenderPass(RenderPass* renderPass, Framebuffer* framebuffer, std::uint32_t imageIndex);
	void nextSubpass(VkSubpassContents subpassContents = VK_SUBPASS_CONTENTS_INLINE);
	void endRenderPass();

	// Pipeline
	void bindGraphicPipeline(VkPipeline pipeline);
	
	// Descriptor sets / push constants
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
	// Gets the VkDescriptorSet handle for the current frame
	VkDescriptorSet& getDescriptorSetHandle(DescriptorSet* descriptorSet);

	// Rendering
	void drawDirect(std::uint32_t vertexCount, std::uint32_t instanceCount, std::uint32_t firstVertex, std::uint32_t firstInstance);
	void drawMesh(MeshData& meshData, const std::function<void(MeshData&)>& perMeshCallback = nullptr);
	void drawMeshGeometry(MeshData& meshData, const std::function<void(MeshData&)>& perMeshCallback = nullptr);
	void drawLineMesh(LineMeshData& lineMeshData);
	void renderImGUI();

	// Blitting
	void blitImage(
		VkImage srcImage,
		VkImageLayout srcImageLayout,
		VkImage dstImage,
		VkImageLayout dstImageLayout,
		std::uint32_t regionCount,
		const VkImageBlit* pRegions,
		VkFilter filter);
	void blitImageToSwapchain(
		VkImage srcImage,
		VkImageLayout srcImageLayout,
		VkImage swapchainImage,
		VkExtent2D renderExtent,
		VkFilter filter);
	
	// Dynamic states
	void setCullMode(VkCullModeFlags cullMode);
	void setDepthBias(float depthBiasConstant, float depthBiasClamp, float depthBiasSlopeFactor);
	void setDepthTestEnable(VkBool32 value);

	// Barriers
	void bufferBarrier(
		VkBuffer buffer,
		VkAccessFlags srcAccessMask,
		VkAccessFlags dstAccessMask,
		VkPipelineStageFlags srcStageMask,
		VkPipelineStageFlags dstStageMask,
		VkDeviceSize size = VK_WHOLE_SIZE,
		VkDeviceSize offset = 0,
		std::uint32_t srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		std::uint32_t dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED);

	void imageBarrier(
		VkImage image,
		VkAccessFlags srcAccessMask,
		VkAccessFlags dstAccessMask,
		VkImageLayout srcLayout,
		VkImageLayout dstLayout,
		VkPipelineStageFlags srcStageMask,
		VkPipelineStageFlags dstStageMask,
		VkImageSubresourceRange range = VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
		std::uint32_t srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		std::uint32_t dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED);

	// Queries
	void resetQueryPool(vk::QueryPool& queryPool, std::uint32_t firstQuery, std::uint32_t queryCount);
	void writeTimestamp(VkPipelineStageFlagBits stageFlag, vk::QueryPool& queryPool, std::uint32_t& query);

	void destroyImGUI();

}