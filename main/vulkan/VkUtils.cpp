#include "VkUtils.hpp"

#include <vector>

#include <cstdio>
#include <cassert>

#include "Error.hpp"
#include "toString.hpp"
#include "VulkanDevice.hpp"

namespace Utils {

	vk::CommandPool createCommandPool(const VulkanWindow& window, VkCommandPoolCreateFlags createFlags) {
		VkCommandPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolInfo.queueFamilyIndex = window.graphicsFamilyIndex;
		poolInfo.flags = createFlags;

		VkCommandPool cpool = VK_NULL_HANDLE;
		if (const auto res = vkCreateCommandPool(window.device->device, &poolInfo, nullptr, &cpool); VK_SUCCESS != res) {
			throw Utils::Error("Unable to create command pool\n vkCreateCommandPool() returned %s", Utils::toString(res).c_str());
		}

		return vk::CommandPool(window.device->device, cpool);
	}

	VkCommandBuffer allocCommandBuffer(const VulkanWindow& window, VkCommandPool cmdPool) {
		VkCommandBufferAllocateInfo cbufInfo{};
		cbufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		cbufInfo.commandPool = cmdPool;
		cbufInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		cbufInfo.commandBufferCount = 1;

		VkCommandBuffer cbuff = VK_NULL_HANDLE;
		if (const auto res = vkAllocateCommandBuffers(window.device->device, &cbufInfo, &cbuff); VK_SUCCESS != res) {
			throw Utils::Error("Unable to allocate command buffers\n vkAllocateCommandBuffers() returned %s", Utils::toString(res).c_str());
		}

		return cbuff;
	}


	vk::Fence createFence(const VulkanWindow& window, VkFenceCreateFlags createFlags) {
		VkFenceCreateInfo fenceInfo{};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceInfo.flags = createFlags;

		VkFence fence = VK_NULL_HANDLE;
		if (const auto res = vkCreateFence(window.device->device, &fenceInfo, nullptr, &fence); VK_SUCCESS != res) {
			throw Utils::Error("Unable to create fence\n vkCreateFence() returned %s", Utils::toString(res).c_str());
		}

		return vk::Fence(window.device->device, fence);
	}

	vk::Semaphore createSemaphore(const VulkanWindow& window) {
		VkSemaphoreCreateInfo semaphoreInfo{};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkSemaphore semaphore = VK_NULL_HANDLE;
		if (const auto res = vkCreateSemaphore(window.device->device, &semaphoreInfo, nullptr, &semaphore); VK_SUCCESS != res)
			throw Utils::Error("Unable to create semaphore\n vkCreateSemaphore() returned %s", Utils::toString(res).c_str());

		return vk::Semaphore(window.device->device, semaphore);
	}

	vk::DescriptorPool createDescriptorPool(const VulkanWindow& window, std::uint32_t maxDescriptors, std::uint32_t maxSets) {
		const VkDescriptorPoolSize pools[] = {
			{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, maxDescriptors },
			{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, maxDescriptors }
		};

		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.maxSets = maxSets;
		poolInfo.poolSizeCount = sizeof(pools) / sizeof(pools[0]);
		poolInfo.pPoolSizes = pools;

		VkDescriptorPool pool = VK_NULL_HANDLE;
		if (const auto res = vkCreateDescriptorPool(window.device->device, &poolInfo, nullptr, &pool); VK_SUCCESS != res)
			throw Utils::Error("Unable to create descriptor pool\n vkCreateDescriptorPool() returned %s", Utils::toString(res).c_str());

		return vk::DescriptorPool(window.device->device, pool);
	}

	VkDescriptorSet allocDescriptorSet(const VulkanWindow& window, VkDescriptorPool descPool, VkDescriptorSetLayout descSetLayout) {
		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = descPool;
		allocInfo.descriptorSetCount = 1;
		allocInfo.pSetLayouts = &descSetLayout;

		VkDescriptorSet dset = VK_NULL_HANDLE;
		if (const auto res = vkAllocateDescriptorSets(window.device->device, &allocInfo, &dset); VK_SUCCESS != res)
			throw Utils::Error("Unable to allocate descriptor set\n vkAllocateDescriptorSets() returned %s", Utils::toString(res).c_str());

		return dset;
	}

	vk::Sampler createDefaultSampler(const VulkanWindow& window) {
		VkSamplerCreateInfo samplerInfo{};
		samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerInfo.magFilter = VK_FILTER_LINEAR;
		samplerInfo.minFilter = VK_FILTER_LINEAR;
		samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.minLod = 0.0f;
		samplerInfo.maxLod = VK_LOD_CLAMP_NONE;
		samplerInfo.mipLodBias = 0.0f;

		VkSampler sampler = VK_NULL_HANDLE;
		if (const auto res = vkCreateSampler(window.device->device, &samplerInfo, nullptr, &sampler); VK_SUCCESS != res)
			throw Utils::Error("Unable to create sampler\n vkCreateSampler() returned %s", Utils::toString(res).c_str());

		return vk::Sampler(window.device->device, sampler);
	}

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
		if (const auto res = vkCreateSampler(window.device->device, &samplerInfo, nullptr, &sampler); VK_SUCCESS != res)
			throw Utils::Error("Unable to create sampler\n vkCreateSampler() returned %s", Utils::toString(res).c_str());

		return vk::Sampler(window.device->device, sampler);
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
		uint32_t dstQueueFamilyIndex
	) {
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
		std::uint32_t dstQueueFamilyIndex
	) {
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


}