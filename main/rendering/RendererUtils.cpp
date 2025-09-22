#include "RendererUtils.hpp"

#include "Error.hpp"
#include "../vulkan/VkUtils.hpp"
#include "../imgui/imgui.h"
#include "../imgui/backends/imgui_impl_vulkan.h"
#include "../imgui/backends/imgui_impl_glfw.h"

namespace RendererUtils {

	VkCommandBuffer boundCommandBuffer = VK_NULL_HANDLE;

	void checkCommandBuffer() {
		if (boundCommandBuffer == VK_NULL_HANDLE) {
			throw Utils::Error("Running Vulkan command without a bound command buffer!\n");
		}
	}

	void bindCommandBuffer(VkCommandBuffer cmdBuff) {
		//std::fprintf(stderr, "Bound command buffer %p\n", cmdBuff);
		boundCommandBuffer = cmdBuff;
	}

	void beginCommandBuffer(VkCommandBufferUsageFlags usageFlags) {
		checkCommandBuffer();

		VkUtils::beginCommandBuffer(boundCommandBuffer, usageFlags);
	}

	void endCommandBuffer() {
		checkCommandBuffer();

		VkUtils::endCommandBuffer(boundCommandBuffer);
	}

	void updateUniformBuffer(_UniformBuffer& uniformBuffer) {
		checkCommandBuffer();

		uniformBuffer->update(boundCommandBuffer);
	}

	void updateShaderStorageBuffer(_ShaderStorageBuffer& shaderStorageBuffer) {
		checkCommandBuffer();

		shaderStorageBuffer->update(boundCommandBuffer);
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

	void endRenderPass() {
		vkCmdEndRenderPass(boundCommandBuffer);
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

	void drawMesh(MeshData& meshData, const std::function<void(MeshData&)>& perMeshCallback) {
		if (perMeshCallback)
			perMeshCallback(meshData);

		VkBuffer vBuffers[4] = {
			meshData.posBuffer.buffer,
			meshData.texCoordBuffer.buffer,
			meshData.normalsBuffer.buffer,
			meshData.tbnFrameBuffer.buffer
		};
		VkBuffer iBuffer = meshData.indicesBuffer.buffer;
		VkDeviceSize vOffsets[4]{};
		VkDeviceSize iOffset{};

		vkCmdBindVertexBuffers(boundCommandBuffer, 0, 4, vBuffers, vOffsets);
		vkCmdBindIndexBuffer(boundCommandBuffer, iBuffer, iOffset, VK_INDEX_TYPE_UINT32);

		vkCmdDrawIndexed(boundCommandBuffer, static_cast<std::uint32_t>(meshData.indicesCount), 1, 0, 0, 0);
	}

	void drawMeshGeometry(MeshData& meshData, const std::function<void(MeshData&)>& perMeshCallback) {
		if (perMeshCallback)
			perMeshCallback(meshData);

		VkBuffer vBuffer = meshData.posBuffer.buffer;
		VkBuffer iBuffer = meshData.indicesBuffer.buffer;
		VkDeviceSize vOffset{};
		VkDeviceSize iOffset{};

		vkCmdBindVertexBuffers(boundCommandBuffer, 0, 1, &vBuffer, &vOffset);
		vkCmdBindIndexBuffer(boundCommandBuffer, iBuffer, iOffset, VK_INDEX_TYPE_UINT32);

		vkCmdDrawIndexed(boundCommandBuffer, static_cast<std::uint32_t>(meshData.indicesCount), 1, 0, 0, 0);
	}

	void drawLineMesh(LineMeshData& lineMeshData) {
		VkBuffer vBuffers[2] = {
			lineMeshData.posBuffer.buffer,
			lineMeshData.colBuffer.buffer,
		};
		VkBuffer iBuffer = lineMeshData.indicesBuffer.buffer;
		VkDeviceSize vOffsets[2]{};
		VkDeviceSize iOffset{};

		vkCmdBindVertexBuffers(boundCommandBuffer, 0, 2, vBuffers, vOffsets);
		vkCmdBindIndexBuffer(boundCommandBuffer, iBuffer, iOffset, VK_INDEX_TYPE_UINT32);

		vkCmdDrawIndexed(boundCommandBuffer, static_cast<std::uint32_t>(lineMeshData.indicesCount), 1, 0, 0, 0);
	}

	void renderImGUI() {
		checkCommandBuffer();

		if (ImGui::GetDrawData() != nullptr)
			ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), boundCommandBuffer);
	}

	void setCullMode(VkCullModeFlags cullMode) {
		checkCommandBuffer();

		vkCmdSetCullMode(boundCommandBuffer, cullMode);
	}

	void setDepthBias(float depthBiasConstant, float depthBiasClamp, float depthBiasSlopeFactor) {
		checkCommandBuffer();

		vkCmdSetDepthBias(boundCommandBuffer, depthBiasConstant, depthBiasClamp, depthBiasSlopeFactor);
	}

	void bufferBarrier(
		VkBuffer buffer, 
		VkAccessFlags srcAccessMask, 
		VkAccessFlags dstAccessMask, 
		VkPipelineStageFlags srcStageMask, 
		VkPipelineStageFlags dstStageMask, 
		VkDeviceSize size, 
		VkDeviceSize offset,
		std::uint32_t srcQueueFamilyIndex, 
		std::uint32_t dstQueueFamilyIndex) 
	{
		checkCommandBuffer();

		VkUtils::bufferBarrier(boundCommandBuffer, buffer, srcAccessMask, dstAccessMask, srcStageMask, dstStageMask, size, offset, srcQueueFamilyIndex, dstQueueFamilyIndex);
	}

	void imageBarrier(
		VkImage image, 
		VkAccessFlags srcAccessMask, 
		VkAccessFlags dstAccessMask, 
		VkImageLayout srcLayout, 
		VkImageLayout dstLayout, 
		VkPipelineStageFlags srcStageMask, 
		VkPipelineStageFlags dstStageMask, 
		VkImageSubresourceRange range, 
		std::uint32_t srcQueueFamilyIndex, 
		std::uint32_t dstQueueFamilyIndex) 
	{
		checkCommandBuffer();

		VkUtils::imageBarrier(boundCommandBuffer, image, srcAccessMask, dstAccessMask, srcLayout, dstLayout, srcStageMask, dstStageMask, range, srcQueueFamilyIndex, dstQueueFamilyIndex);
	}

	void destroyImGUI() {
		ImGui_ImplVulkan_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();
	}

}