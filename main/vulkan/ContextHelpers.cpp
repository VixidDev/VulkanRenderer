#include "ContextHelpers.hpp"

#include "Error.hpp"
#include "toString.hpp"

namespace Utils {

	std::unordered_set<std::string> getInstanceLayers() {
		std::uint32_t numLayers = 0;
		if (const auto res = vkEnumerateInstanceLayerProperties(&numLayers, nullptr); VK_SUCCESS != res) {
			throw Utils::Error("Unable to enumerate layers\n"
				"vkEnumerateInstanceLayerProperties() returned %s\n", Utils::toString(res).c_str()
			);
		}

		std::vector<VkLayerProperties> layers(numLayers);
		if (const auto res = vkEnumerateInstanceLayerProperties(&numLayers, layers.data()); VK_SUCCESS != res) {
			throw Utils::Error("Unable to get layer properties\n"
				"vkEnumerateInstanceLayerProperties() returned %s", Utils::toString(res).c_str()
			);
		}

		std::unordered_set<std::string> res;
		for (const auto& layer : layers)
			res.insert(layer.layerName);

		return res;
	}

	std::unordered_set<std::string> getInstanceExtensions() {
		std::uint32_t numExtensions = 0;
		if (const auto res = vkEnumerateInstanceExtensionProperties(nullptr, &numExtensions, nullptr); VK_SUCCESS != res) {	
			throw Utils::Error("Unable to enumerate extensions\n"
				"vkEnumerateInstanceExtensionProperties() returned %s", Utils::toString(res).c_str() 
			);
		}

		std::vector<VkExtensionProperties> extensions(numExtensions);
		if (const auto res = vkEnumerateInstanceExtensionProperties(nullptr, &numExtensions, extensions.data()); VK_SUCCESS != res) {	
			throw Utils::Error("Unable to get extension properties\n" 
				"vkEnumerateInstanceExtensionProperties() returned %s", Utils::toString(res).c_str() 
			);
		}

		std::unordered_set<std::string> res;
		for (const auto& extension : extensions)
			res.insert(extension.extensionName);

		return res;
	}

	std::unordered_set<std::string> getDeviceExtensions(VkPhysicalDevice physicalDevice) {
		std::uint32_t extensionCount = 0;
		if (const auto res = vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, nullptr); VK_SUCCESS != res) {
			throw Utils::Error("Unable to get device extension count\n"
				"vkEnumerateDeviceExtensionProperties() returned %s", Utils::toString(res).c_str()
			);
		}

		std::vector<VkExtensionProperties> extensions(extensionCount);
		if (const auto res = vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, extensions.data()); VK_SUCCESS != res) {
			throw Utils::Error("Unable to get device extensions\n"
				"vkEnumerateDeviceExtensionProperties() returned %s", Utils::toString(res).c_str()
			);
		}

		std::unordered_set<std::string> ret;
		for (const auto& ext : extensions)
			ret.emplace(ext.extensionName);

		return ret;
	}

	VKAPI_ATTR VkBool32 VKAPI_CALL debugUtilCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT type, const VkDebugUtilsMessengerCallbackDataEXT* data, void* /*aUserPtr*/) {
		if (1461184347 == data->messageIdNumber)
			return VK_FALSE;

		std::fprintf(stderr, "%s (%s): %s (%d)\n%s\n--\n", Utils::toString(severity).c_str(), messageTypeFlags(type).c_str(), data->pMessageIdName, data->messageIdNumber, data->pMessage);

		return VK_FALSE;
	}

}