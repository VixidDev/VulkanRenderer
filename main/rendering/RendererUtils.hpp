#pragma once

#include "Renderer.hpp"

namespace RendererUtils {

	void checkCommandBuffer();

	// Command buffer
	void bindCommandBuffer(VkCommandBuffer cmdBuff);
	void beginCommandBuffer(VkCommandBufferUsageFlags usageFlags = 0);
	void endCommandBuffer();

	// Buffers
	void updateUniformBuffer(_UniformBuffer& uniformBuffer);
	void updateShaderStorageBuffer(_ShaderStorageBuffer& shaderStorageBuffer);

	// Render passes
	void beginRenderPass(RenderPass* renderPass, Framebuffer* framebuffer, std::uint32_t imageIndex);
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

	// Rendering
	void drawMesh(MeshData& meshData, const std::function<void(MeshData&)>& perMeshCallback = nullptr);
	void drawMeshGeometry(MeshData& meshData, const std::function<void(MeshData&)>& perMeshCallback = nullptr);
	void drawLineMesh(LineMeshData& lineMeshData);
	void renderImGUI();
	
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

	void destroyImGUI();

}