#pragma once

#include <volk/volk.h>

#include <string>
#include <vector>
#include <unordered_set>

namespace Utils {

	std::unordered_set<std::string> getInstanceLayers();
	std::unordered_set<std::string> getInstanceExtensions();

	VkInstance createInstance(
		const std::vector<const char*>& enabledLayers = {},
		const std::vector<const char*>& enabledInstanceExtensions = {},
		bool enableDebugUtils = false
	);

	std::unordered_set<std::string> getDeviceExtensions(VkPhysicalDevice physicalDevice);

	VkDebugUtilsMessengerEXT createDebugMessenger(VkInstance instance);
	VKAPI_ATTR VkBool32 VKAPI_CALL debugUtilCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT type, const VkDebugUtilsMessengerCallbackDataEXT* data, void* userPtr);

}
