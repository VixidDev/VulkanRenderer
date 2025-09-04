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

	VkInstance createInstance(const std::vector<const char*>& enabledLayers, const std::vector<const char*>& enabledExtensions, bool enableDebugUtils) {
		// Prepare debug messenger info
		VkDebugUtilsMessengerCreateInfoEXT debugInfo{};

		if (enableDebugUtils) {
			debugInfo.sType  = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
			debugInfo.messageSeverity  = /*VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT | */VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
			debugInfo.messageType      = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
			debugInfo.pfnUserCallback  = &debugUtilCallback;
			debugInfo.pUserData        = nullptr;
		}

		// Prepare application info
		// The `apiVersion` is the *highest* version of Vulkan than the
		// application can use. We can therefore safely set it to 1.3, even if
		// we are not intending to use any 1.3 features (and want to run on
		// pre-1.3 implementations).
		VkApplicationInfo appInfo{};
		appInfo.sType  = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName    = "VulkanRenderer";
		appInfo.applicationVersion  = 1;
		appInfo.apiVersion          = VK_MAKE_API_VERSION( 0, 1, 3, 0 ); // Version 1.3

		// Create instance
		VkInstanceCreateInfo instanceInfo{};
		instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;

		instanceInfo.enabledLayerCount        = std::uint32_t(enabledLayers.size());
		instanceInfo.ppEnabledLayerNames      = enabledLayers.data();

		instanceInfo.enabledExtensionCount    = std::uint32_t(enabledExtensions.size());
		instanceInfo.ppEnabledExtensionNames  = enabledExtensions.data();

		instanceInfo.pApplicationInfo = &appInfo;

		if (enableDebugUtils) {
			debugInfo.pNext = instanceInfo.pNext;
			instanceInfo.pNext = &debugInfo; 
		}

		VkInstance instance;
		if (const auto res = vkCreateInstance(&instanceInfo, nullptr, &instance); VK_SUCCESS != res) {
			throw Utils::Error("Unable to create Vulkan instance\n"
				"vkCreateInstance() returned %s", Utils::toString(res).c_str() 
			);
		}

		return instance;
	}

	VkDebugUtilsMessengerEXT createDebugMessenger(VkInstance instance) {
		// Set up the debug messaging for the rest of the application
		VkDebugUtilsMessengerCreateInfoEXT debugInfo{};
		debugInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		debugInfo.messageSeverity = /*VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT | */ VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		debugInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		debugInfo.pfnUserCallback = &debugUtilCallback;
		debugInfo.pUserData = nullptr;

		VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
		if (const auto res = vkCreateDebugUtilsMessengerEXT(instance, &debugInfo, nullptr, &messenger); VK_SUCCESS != res) {
			throw Utils::Error("Unable to set up debug messenger\n"
				"vkCreateDebugUtilsMessengerEXT() returned %s", Utils::toString(res).c_str()
			);
		}

		return messenger;
	}

	VKAPI_ATTR VkBool32 VKAPI_CALL debugUtilCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT type, const VkDebugUtilsMessengerCallbackDataEXT* data, void* /*aUserPtr*/) {
		if (1461184347 == data->messageIdNumber)
			return VK_FALSE;

		std::fprintf(stderr, "%s (%s): %s (%d)\n%s\n--\n", Utils::toString(severity).c_str(), messageTypeFlags(type).c_str(), data->pMessageIdName, data->messageIdNumber, data->pMessage);

		return VK_FALSE;
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

}