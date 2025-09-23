#include "VulkanDevice.hpp"

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

	if (const auto res = vkCreateCommandPool(this->device, &poolInfo, nullptr, &this->cmdPool); VK_SUCCESS != res) {
		throw Utils::Error("Unable to create command pool\n vkCreateCommandPool() returned %s", Utils::toString(res).c_str());
	}
}

void VulkanDevice::createDescriptorPool() {
	const VkDescriptorPoolSize pools[] = {
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2048 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 100 },
		{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2048 },
		{ VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 2048 }
	};

	VkDescriptorPoolCreateInfo poolInfo{};
	poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.maxSets = 1024;
	poolInfo.poolSizeCount = sizeof(pools) / sizeof(pools[0]);
	poolInfo.pPoolSizes = pools;
	poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

	if (const auto res = vkCreateDescriptorPool(this->device, &poolInfo, nullptr, &this->descPool); VK_SUCCESS != res)
		throw Utils::Error("Unable to create descriptor pool\n vkCreateDescriptorPool() returned %s", Utils::toString(res).c_str());
}

VkSampleCountFlagBits VulkanDevice::getSampleCount(std::size_t index) {
	return this->possibleSampleCounts[index];
}