/*
* File credits to Markus Billeter
*/
#pragma once

#include <volk/volk.h>

#include <string>

#include <cstdint>

namespace Utils {

	std::string toString(VkResult result);
	std::string toString(VkPhysicalDeviceType deviceType);
	std::string toString(VkDebugUtilsMessageSeverityFlagBitsEXT severity);

	std::string queueFlags(VkQueueFlags flags);
	std::string messageTypeFlags(VkDebugUtilsMessageTypeFlagsEXT flags);
	std::string memoryHeapFlags(VkMemoryHeapFlags flags);
	std::string memoryPropertyFlags(VkMemoryPropertyFlags flags);

	std::string driverVersion(std::uint32_t vendorId, std::uint32_t driverVersion);
}
