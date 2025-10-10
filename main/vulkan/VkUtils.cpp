#include "VkUtils.hpp"

#include <cstdio>
#include <cassert>
#include <vector>
#include <limits>

#include "Error.hpp"
#include "toString.hpp"
#include "VulkanDevice.hpp"
#include "Swapchain.hpp"

namespace VkUtils {

	VkCommandBuffer createCommandBuffer(const VulkanWindow& window, VkCommandPool cmdPool) {
		VkCommandBufferAllocateInfo cbufInfo{};
		cbufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		cbufInfo.commandPool = cmdPool;
		cbufInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		cbufInfo.commandBufferCount = 1;

		VkCommandBuffer cbuff = VK_NULL_HANDLE;
		if (const auto res = vkAllocateCommandBuffers(window.getDevice()->getDevice(), &cbufInfo, &cbuff); VK_SUCCESS != res) {
			throw Utils::Error("Unable to allocate command buffers\n vkAllocateCommandBuffers() returned %s", Utils::toString(res).c_str());
		}

		return cbuff;
	}

	void beginCommandBuffer(VkCommandBuffer cmdBuff, VkCommandBufferUsageFlags usageFlags) {
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = usageFlags;
		beginInfo.pInheritanceInfo = nullptr;

		if (const auto res = vkBeginCommandBuffer(cmdBuff, &beginInfo); VK_SUCCESS != res)
			throw Utils::Error("Unable to begin command buffer\n vkBeginCommandBuffer() returned %s", Utils::toString(res).c_str());
	}

	void endCommandBuffer(VkCommandBuffer cmdBuff) {
		if (const auto res = vkEndCommandBuffer(cmdBuff); VK_SUCCESS != res)
			throw Utils::Error("Unable to end command buffer\n vkEndCommandBuffer() returned %s", Utils::toString(res).c_str());
	}

	void endAndSubmitCommandBuffer(const VulkanWindow& window, VkCommandBuffer cmdBuff) {
		endCommandBuffer(cmdBuff);

		vk::Fence uploadComplete = createFence(window);

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &cmdBuff;

		if (const auto res = vkQueueSubmit(window.getDevice()->getGraphicsQueue(), 1, &submitInfo, uploadComplete.handle); VK_SUCCESS != res)
			throw Utils::Error("Unable to queue submit\n vkQueueSubmit() returned %s", Utils::toString(res).c_str());

		if (const auto res = vkWaitForFences(window.getDevice()->getDevice(), 1, &uploadComplete.handle, VK_TRUE, std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res)
			throw Utils::Error("Unable to wait for fences\n vkWaitForFences() returned %s", Utils::toString(res).c_str());

		vkFreeCommandBuffers(window.getDevice()->getDevice(), window.getDevice()->getCmdPool(), 1, &cmdBuff);
	}

	vk::Fence createFence(const VulkanWindow& window, VkFenceCreateFlags createFlags) {
		VkFenceCreateInfo fenceInfo{};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceInfo.flags = createFlags;

		VkFence fence = VK_NULL_HANDLE;
		if (const auto res = vkCreateFence(window.getDevice()->getDevice(), &fenceInfo, nullptr, &fence); VK_SUCCESS != res) {
			throw Utils::Error("Unable to create fence\n vkCreateFence() returned %s", Utils::toString(res).c_str());
		}

		return vk::Fence(window.getDevice()->getDevice(), fence);
	}

	vk::Semaphore createSemaphore(const VulkanWindow& window) {
		VkSemaphoreCreateInfo semaphoreInfo{};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkSemaphore semaphore = VK_NULL_HANDLE;
		if (const auto res = vkCreateSemaphore(window.getDevice()->getDevice(), &semaphoreInfo, nullptr, &semaphore); VK_SUCCESS != res)
			throw Utils::Error("Unable to create semaphore\n vkCreateSemaphore() returned %s", Utils::toString(res).c_str());

		return vk::Semaphore(window.getDevice()->getDevice(), semaphore);
	}

	void waitForFences(const VulkanWindow& window, std::vector<vk::Fence>& fences, std::size_t frameIndex) {
		if (const auto res = vkWaitForFences(window.getDevice()->getDevice(), 1, &fences[frameIndex].handle, VK_TRUE, std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res)
			throw Utils::Error("Unable to wait for frame fence %u\n vkWaitForFences() returned %s", frameIndex, Utils::toString(res).c_str());
	}

	void resetFences(const VulkanWindow& window, std::vector<vk::Fence>& fences, std::size_t frameIndex) {
		if (const auto res = vkResetFences(window.getDevice()->getDevice(), 1, &fences[frameIndex].handle); VK_SUCCESS != res)
			throw Utils::Error("Unable to reset frame fence %u\n vkResetFences() returned %s", frameIndex, Utils::toString(res).c_str());
	}

	VkResult acquireNextSwapchainImage(const VulkanWindow& window, std::vector<vk::Semaphore>& semaphores, std::size_t frameIndex, std::uint32_t& imageIndex) {
		const VkResult acquireResult = vkAcquireNextImageKHR(
			window.getDevice()->getDevice(),
			window.getSwapchain()->getHandle(),
			std::numeric_limits<std::uint64_t>::max(),
			semaphores[frameIndex].handle,
			VK_NULL_HANDLE,
			&imageIndex
		);

		if (acquireResult == VK_SUBOPTIMAL_KHR || acquireResult == VK_ERROR_OUT_OF_DATE_KHR)
			return acquireResult;

		if (acquireResult != VK_SUCCESS)
			throw Utils::Error("Unable to acquire next swapchain image\n vkAcquireNextImageKHR() returned %s", Utils::toString(acquireResult).c_str());

		return acquireResult;
	}

	VkDescriptorSet createDescriptorSet(const VulkanWindow& window, VkDescriptorPool descPool, VkDescriptorSetLayout descSetLayout) {
		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = descPool;
		allocInfo.descriptorSetCount = 1;
		allocInfo.pSetLayouts = &descSetLayout;

		VkDescriptorSet dset = VK_NULL_HANDLE;
		if (const auto res = vkAllocateDescriptorSets(window.getDevice()->getDevice(), &allocInfo, &dset); VK_SUCCESS != res)
			throw Utils::Error("Unable to allocate descriptor set\n vkAllocateDescriptorSets() returned %s", Utils::toString(res).c_str());

		return dset;
	}

	vk::Sampler createTextureSampler(const VulkanWindow& window, SamplerInfo samplerInfo) {
		bool deviceEnabledAnisotropy = window.getDevice()->getDeviceFeatures().features.samplerAnisotropy;

		VkSamplerCreateInfo samplerCreateInfo{};
		samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerCreateInfo.magFilter = samplerInfo.magFilter;
		samplerCreateInfo.minFilter = samplerInfo.minFilter;
		samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		samplerCreateInfo.addressModeU = samplerInfo.addressMode;
		samplerCreateInfo.addressModeV = samplerInfo.addressMode;
		samplerCreateInfo.addressModeW = samplerInfo.addressMode;
		samplerCreateInfo.mipLodBias = 0.0f;
		samplerCreateInfo.anisotropyEnable = deviceEnabledAnisotropy ? samplerInfo.anisotropyEnable : VK_FALSE;
		samplerCreateInfo.maxAnisotropy = samplerInfo.maxAnisotropy;
		samplerCreateInfo.compareEnable = samplerInfo.compareEnable;
		samplerCreateInfo.compareOp = samplerInfo.compareOp;
		samplerCreateInfo.minLod = 0.0f;
		samplerCreateInfo.maxLod = VK_LOD_CLAMP_NONE;
		samplerCreateInfo.borderColor = samplerInfo.borderColor;

		VkSampler sampler = VK_NULL_HANDLE;
		if (const auto res = vkCreateSampler(window.getDevice()->getDevice(), &samplerCreateInfo, nullptr, &sampler); VK_SUCCESS != res)
			throw Utils::Error("Unable to create sampler\n vkCreateSampler() returned %s", Utils::toString(res).c_str());

		return vk::Sampler(window.getDevice()->getDevice(), sampler);
	}

	// TODO: Remove
	vk::Sampler createDefaultSampler(const VulkanWindow& window) {
		VkSamplerCreateInfo samplerInfo{};
		samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerInfo.magFilter = VK_FILTER_LINEAR;
		samplerInfo.minFilter = VK_FILTER_LINEAR;
		samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.minLod = 0.0f;
		samplerInfo.maxLod = VK_LOD_CLAMP_NONE;
		samplerInfo.mipLodBias = 0.0f;

		VkSampler sampler = VK_NULL_HANDLE;
		if (const auto res = vkCreateSampler(window.getDevice()->getDevice(), &samplerInfo, nullptr, &sampler); VK_SUCCESS != res)
			throw Utils::Error("Unable to create sampler\n vkCreateSampler() returned %s", Utils::toString(res).c_str());

		return vk::Sampler(window.getDevice()->getDevice(), sampler);
	}

	// TODO: Remove
	vk::Sampler createShadowSampler(const VulkanWindow& window) {
		VkSamplerCreateInfo samplerInfo{};
		samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerInfo.magFilter = VK_FILTER_LINEAR;
		samplerInfo.minFilter = VK_FILTER_LINEAR;
		samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		samplerInfo.compareEnable = VK_TRUE;
		samplerInfo.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		samplerInfo.minLod = 0.0f;
		samplerInfo.maxLod = VK_LOD_CLAMP_NONE;
		samplerInfo.mipLodBias = 0.0f;

		VkSampler sampler = VK_NULL_HANDLE;
		if (const auto res = vkCreateSampler(window.getDevice()->getDevice(), &samplerInfo, nullptr, &sampler); VK_SUCCESS != res)
			throw Utils::Error("Unable to create sampler\n vkCreateSampler() returned %s", Utils::toString(res).c_str());

		return vk::Sampler(window.getDevice()->getDevice(), sampler);
	}

	void bufferBarrier(
		VkCommandBuffer cmdBuff,
		VkBuffer buffer,
		VkAccessFlags srcAccessMask,
		VkAccessFlags dstAccessMask,
		VkPipelineStageFlags srcStageMask,
		VkPipelineStageFlags dstStageMask,
		VkDeviceSize size,
		VkDeviceSize offset,
		uint32_t srcQueueFamilyIndex,
		uint32_t dstQueueFamilyIndex) {
		VkBufferMemoryBarrier bbarrier{};
		bbarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
		bbarrier.srcAccessMask = srcAccessMask;
		bbarrier.dstAccessMask = dstAccessMask;
		bbarrier.buffer = buffer;
		bbarrier.size = size;
		bbarrier.offset = offset;
		bbarrier.srcQueueFamilyIndex = srcQueueFamilyIndex;
		bbarrier.dstQueueFamilyIndex = dstQueueFamilyIndex;

		vkCmdPipelineBarrier(
			cmdBuff,
			srcStageMask,
			dstStageMask,
			0,
			0,
			nullptr,
			1,
			&bbarrier,
			0,
			nullptr
		);
	}

	void imageBarrier(
		VkCommandBuffer cmdBuff,
		VkImage image,
		VkAccessFlags srcAccessMask,
		VkAccessFlags dstAccessMask,
		VkImageLayout srcLayout,
		VkImageLayout dstLayout,
		VkPipelineStageFlags srcStageMask,
		VkPipelineStageFlags dstStageMask,
		VkImageSubresourceRange range,
		std::uint32_t srcQueueFamilyIndex,
		std::uint32_t dstQueueFamilyIndex) {
		VkImageMemoryBarrier ibarrier{};
		ibarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		ibarrier.image = image;
		ibarrier.srcAccessMask = srcAccessMask;
		ibarrier.dstAccessMask = dstAccessMask;
		ibarrier.srcQueueFamilyIndex = srcQueueFamilyIndex;
		ibarrier.dstQueueFamilyIndex = dstQueueFamilyIndex;
		ibarrier.oldLayout = srcLayout;
		ibarrier.newLayout = dstLayout;
		ibarrier.subresourceRange = range;

		vkCmdPipelineBarrier(cmdBuff, srcStageMask, dstStageMask, 0, 0, nullptr, 0, nullptr, 1, &ibarrier);
	}

	vk::QueryPool createQueryPool(const VulkanWindow& window, VkQueryType queryType, std::uint32_t queryCount) {
		VkQueryPoolCreateInfo queryPoolInfo = {
			.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
			.queryType = queryType,
			.queryCount = queryCount
		};

		VkQueryPool queryPool = VK_NULL_HANDLE;
		if (const auto res = vkCreateQueryPool(window.getDevice()->getDevice(), &queryPoolInfo, nullptr, &queryPool); VK_SUCCESS != res)
			throw Utils::Error("Unable to create query pool\nvkCreateQueryPool(): returned %s\n", Utils::toString(res).c_str());

		return vk::QueryPool(window.getDevice()->getDevice(), queryPool);
	}

	void getQueryPoolResults(
		const VulkanWindow& window,
		vk::QueryPool& queryPool,
		std::vector<std::uint64_t>& queryResults,
		std::uint32_t queryCount,
		VkQueryResultFlags resultFlags) {
		const VkResult res = vkGetQueryPoolResults(
			window.getDevice()->getDevice(),
			queryPool.handle,
			0, queryCount,
			queryCount * sizeof(std::uint64_t),
			queryResults.data(),
			sizeof(std::uint64_t),
			resultFlags
		);

		if (res == VK_NOT_READY) {
			std::fprintf(stderr, "vkGetQueryPoolResults(): returned VK_NOT_READY. Are you sure you passed the correct flags?\n");
		} else if (res != VK_SUCCESS) {
			throw Utils::Error("Unable to get query pool results\nvkGetQueryPoolResults(): returned '%s'\n", Utils::toString(res).c_str());
		}
	}

}