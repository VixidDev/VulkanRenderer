#include "VulkanAllocator.hpp"

#include <utility>
#include <cassert>

#include "Error.hpp"
#include "toString.hpp"
#include "VulkanWindow.hpp"
#include "VulkanDevice.hpp"

VulkanAllocator::VulkanAllocator(VulkanWindow* window) {
	VkPhysicalDeviceProperties props{};
	vkGetPhysicalDeviceProperties(window->getDevice()->getPhysicalDevice(), &props);

	VmaVulkanFunctions functions{};
	functions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
	functions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

	VmaAllocatorCreateInfo allocInfo{};
	allocInfo.vulkanApiVersion = props.apiVersion;
	allocInfo.physicalDevice = window->getDevice()->getPhysicalDevice();
	allocInfo.device = window->getDevice()->getDevice();
	allocInfo.instance = window->getInstance();
	allocInfo.pVulkanFunctions = &functions;

	if (const VkResult res = vmaCreateAllocator(&allocInfo, &this->allocator); VK_SUCCESS != res)
		throw Utils::Error("Unable to create allocator\nvmaCreateAllocator() returned %s", Utils::toString(res).c_str());
}

VulkanAllocator::~VulkanAllocator() {
	if (this->allocator != VK_NULL_HANDLE) {
		vmaDestroyAllocator(this->allocator);
	}
}

VulkanAllocator::VulkanAllocator(VmaAllocator aAllocator) noexcept
	: allocator(aAllocator) {}

VulkanAllocator::VulkanAllocator(VulkanAllocator&& aOther) noexcept
	: allocator(std::exchange(aOther.allocator, VK_NULL_HANDLE)) {}

VulkanAllocator& VulkanAllocator::operator=(VulkanAllocator&& aOther) noexcept {
	std::swap(allocator, aOther.allocator);
	return *this;
}
