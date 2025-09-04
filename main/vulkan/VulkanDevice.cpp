#include "VulkanDevice.hpp"

#include <utility>
#include <limits>

#include "Error.hpp"
#include "toString.hpp"
#include "VulkanWindow.hpp"

VulkanDevice::VulkanDevice(VkDevice device, const VulkanWindow& window) noexcept
	: device(device) {
	this->createCommandPool(window);
	this->createDescriptorPool();
}

VulkanDevice::VulkanDevice(VulkanDevice&& aOther) noexcept
	: device(std::exchange(aOther.device, VK_NULL_HANDLE))
	, cmdPool(std::exchange(aOther.cmdPool, VK_NULL_HANDLE))
	, descPool(std::exchange(aOther.descPool, VK_NULL_HANDLE)) {}

VulkanDevice& VulkanDevice::operator=(VulkanDevice&& aOther) noexcept {
	std::swap(device, aOther.device);
	std::swap(cmdPool, aOther.cmdPool);
	std::swap(descPool, aOther.descPool);
	return *this;
}

void VulkanDevice::createCommandPool(const VulkanWindow& window) {
	VkCommandPoolCreateInfo poolInfo{};
	poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	poolInfo.queueFamilyIndex = window.graphicsFamilyIndex;
	poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

	if (const auto res = vkCreateCommandPool(device, &poolInfo, nullptr, &cmdPool); VK_SUCCESS != res) {
		throw Utils::Error("Unable to create command pool\n vkCreateCommandPool() returned %s", Utils::toString(res).c_str());
	}
}

void VulkanDevice::createDescriptorPool() {
	const VkDescriptorPoolSize pools[] = {
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2048 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 100 },
		{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2048 }
	};

	VkDescriptorPoolCreateInfo poolInfo{};
	poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.maxSets = 1024;
	poolInfo.poolSizeCount = sizeof(pools) / sizeof(pools[0]);
	poolInfo.pPoolSizes = pools;
	poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

	if (const auto res = vkCreateDescriptorPool(device, &poolInfo, nullptr, &descPool); VK_SUCCESS != res)
		throw Utils::Error("Unable to create descriptor pool\n vkCreateDescriptorPool() returned %s", Utils::toString(res).c_str());
}

VkSampleCountFlagBits VulkanDevice::getSampleCount(std::size_t index) {
	return this->possibleSampleCounts[index];
}

VkCommandBuffer createCommandBuffer(const VulkanWindow& window) {
	VkCommandBufferAllocateInfo cbufInfo{};
	cbufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	cbufInfo.commandPool = window.device->cmdPool;
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

VkDescriptorSet allocateDescriptorSet(const VulkanWindow& window, VkDescriptorPool descPool, VkDescriptorSetLayout descSetLayout) {
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

void beginCommandBuffer(VkCommandBuffer cmdBuff, VkCommandBufferUsageFlags usageFlags) {
	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = usageFlags;
	beginInfo.pInheritanceInfo = nullptr;

	if (const auto res = vkBeginCommandBuffer(cmdBuff, &beginInfo); VK_SUCCESS != res)
		throw Utils::Error("Unable to begin command buffer\n vkBeginCommandBuffer() returned %s", Utils::toString(res).c_str());
}

void endCommandBuffer(const VulkanWindow& window, VkCommandBuffer cmdBuff) {
	if (const auto res = vkEndCommandBuffer(cmdBuff); VK_SUCCESS != res)
		throw Utils::Error("Unable to end command buffer\n vkEndCommandBuffer() returned %s", Utils::toString(res).c_str());
}

void endAndSubmitCommandBuffer(const VulkanWindow& window, VkCommandBuffer cmdBuff) {
	endCommandBuffer(window, cmdBuff);

	vk::Fence uploadComplete = createFence(window);

	VkSubmitInfo submitInfo{};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &cmdBuff;

	if (const auto res = vkQueueSubmit(window.graphicsQueue, 1, &submitInfo, uploadComplete.handle); VK_SUCCESS != res)
		throw Utils::Error("Unable to queue submit\n vkQueueSubmit() returned %s", Utils::toString(res).c_str());

	if (const auto res = vkWaitForFences(window.device->device, 1, &uploadComplete.handle, VK_TRUE, std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res)
		throw Utils::Error("Unable to wait for fences\n vkWaitForFences() returned %s", Utils::toString(res).c_str());

	vkFreeCommandBuffers(window.device->device, window.device.get()->cmdPool, 1, &cmdBuff);
}

vk::Sampler createTextureSampler(const VulkanWindow& window, SamplerInfo samplerInfo) {
	VkSamplerCreateInfo samplerCreateInfo{};
	samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	samplerCreateInfo.magFilter = samplerInfo.magFilter;
	samplerCreateInfo.minFilter = samplerInfo.minFilter;
	samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	samplerCreateInfo.addressModeU = samplerInfo.addressModeU;
	samplerCreateInfo.addressModeV = samplerInfo.addressModeV;
	samplerCreateInfo.addressModeW = samplerInfo.addressModeW;
	samplerCreateInfo.compareEnable = samplerInfo.compareEnable;
	samplerCreateInfo.compareOp = samplerInfo.compareOp;
	samplerCreateInfo.anisotropyEnable = window.deviceFeatures.samplerAnisotropy;
	samplerCreateInfo.maxAnisotropy = 8.0f;
	samplerCreateInfo.minLod = 0.0f;
	samplerCreateInfo.maxLod = VK_LOD_CLAMP_NONE;
	samplerCreateInfo.mipLodBias = 0.0f;
	samplerCreateInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;

	VkSampler sampler = VK_NULL_HANDLE;
	if (const auto res = vkCreateSampler(window.device->device, &samplerCreateInfo, nullptr, &sampler); VK_SUCCESS != res)
		throw Utils::Error("Unable to create sampler\n vkCreateSampler() returned %s", Utils::toString(res).c_str());

	return vk::Sampler(window.device->device, sampler);
}

void waitForFences(const VulkanWindow& window, std::vector<vk::Fence>& fences, std::size_t frameIndex) {
	if (const auto res = vkWaitForFences(window.device->device, 1, &fences[frameIndex].handle, VK_TRUE, std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res)
		throw Utils::Error("Unable to wait for frame fence %u\n vkWaitForFences() returned %s", frameIndex, Utils::toString(res).c_str());
}

void resetFences(const VulkanWindow& window, std::vector<vk::Fence>& fences, std::size_t frameIndex) {
	if (const auto res = vkResetFences(window.device->device, 1, &fences[frameIndex].handle); VK_SUCCESS != res)
		throw Utils::Error("Unable to reset frame fence %u\n vkResetFences() returned %s", frameIndex, Utils::toString(res).c_str());
}

VkResult acquireNextSwapchainImage(const VulkanWindow& window, std::vector<vk::Semaphore>& semaphores, std::size_t frameIndex, std::uint32_t& imageIndex) {
	const VkResult acquireResult = vkAcquireNextImageKHR(
		window.device->device,
		window.swapchain,
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