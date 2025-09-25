#include "VulkanAllocator.hpp"

#include <utility>
#include <cassert>

#include "Error.hpp"
#include "toString.hpp"
#include "VulkanDevice.hpp"

VulkanAllocator::VulkanAllocator() noexcept = default;

VulkanAllocator::~VulkanAllocator() {
	if (VK_NULL_HANDLE != allocator) {
		vmaDestroyAllocator(allocator);
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

std::unique_ptr<VulkanAllocator> initialiseVulkanAllocator(const VulkanWindow& window) {
	VkPhysicalDeviceProperties props{};
	vkGetPhysicalDeviceProperties(window.physicalDevice, &props);

	VmaVulkanFunctions functions{};
	functions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
	functions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

	VmaAllocatorCreateInfo allocInfo{};
	allocInfo.vulkanApiVersion = props.apiVersion;
	allocInfo.physicalDevice = window.physicalDevice;
	allocInfo.device = window.device->device;
	allocInfo.instance = window.instance;
	allocInfo.pVulkanFunctions = &functions;

	VmaAllocator allocator = VK_NULL_HANDLE;
	if (auto const res = vmaCreateAllocator(&allocInfo, &allocator); VK_SUCCESS != res) {
		throw Utils::Error("Unable to create allocator\n vmaCreateAllocator() returned %s", Utils::toString(res).c_str());
	}

	return std::make_unique<VulkanAllocator>(allocator);
}

