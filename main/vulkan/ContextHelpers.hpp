#pragma once

#include <volk/volk.h>

#include <string>
#include <vector>
#include <unordered_set>

namespace Utils {

	std::unordered_set<std::string> getInstanceLayers();
	std::unordered_set<std::string> getInstanceExtensions();
	std::unordered_set<std::string> getDeviceExtensions(VkPhysicalDevice physicalDevice);

	VKAPI_ATTR VkBool32 VKAPI_CALL debugUtilCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT type, const VkDebugUtilsMessengerCallbackDataEXT* data, void* userPtr);

}
