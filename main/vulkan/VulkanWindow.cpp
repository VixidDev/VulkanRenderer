#include "VulkanWindow.hpp"

#include <tuple>
#include <limits>
#include <vector>
#include <utility>
#include <optional>
#include <algorithm>
#include <unordered_set>

#include <iostream>

#include <cstdio>
#include <cassert>
#include <vulkan/vulkan_core.h>

#include "Error.hpp"
#include "toString.hpp"
#include "ContextHelpers.hpp"
#include "VulkanDevice.hpp"

#if !defined(GLFW_INCLUDE_NONE)
#	define GLFW_INCLUDE_NONE 1
#endif
#include <GLFW/glfw3.h>

VulkanWindow::~VulkanWindow() {
	// Device-related objects
	for (const auto view : swapViews)
		vkDestroyImageView(device->device, view, nullptr);

	if (VK_NULL_HANDLE != swapchain)
		vkDestroySwapchainKHR(device->device, swapchain, nullptr);

	// Window and related objects
	if (VK_NULL_HANDLE != surface)
		vkDestroySurfaceKHR(instance, surface, nullptr);

	if (window) {
		glfwDestroyWindow(window);

		// The following assumes that we never create more than one window;
		// if there are multiple windows, destroying one of them would
		// unload the whole GLFW library. Nevertheless, this solution is
		// convenient when only dealing with one window (which we will do
		// in the exercises), as it ensure that GLFW is unloaded after all
		// window-related resources are.
		glfwTerminate();
	}

	if (device->descPool != VK_NULL_HANDLE)
		vkDestroyDescriptorPool(device->device, device->descPool, nullptr);

	if (device->cmdPool != VK_NULL_HANDLE)
		vkDestroyCommandPool(device->device, device->cmdPool, nullptr);

	if (device->device != VK_NULL_HANDLE)
		vkDestroyDevice(device->device, nullptr);

	if (debugMessenger != VK_NULL_HANDLE)
		vkDestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);

	if (instance != VK_NULL_HANDLE)
		vkDestroyInstance(instance, nullptr);
}

VulkanWindow::VulkanWindow(VulkanWindow&& other) noexcept
	: instance(std::exchange(other.instance, VK_NULL_HANDLE))
	, physicalDevice(std::exchange(other.physicalDevice, VK_NULL_HANDLE))
	, device(std::exchange(other.device, nullptr))
	, graphicsFamilyIndex(other.graphicsFamilyIndex)
	, graphicsQueue(std::exchange(other.graphicsQueue, VK_NULL_HANDLE))
	, presentFamilyIndex(other.presentFamilyIndex)
	, presentQueue(std::exchange(other.presentQueue, VK_NULL_HANDLE))
	, debugMessenger(std::exchange(other.debugMessenger, VK_NULL_HANDLE))
	, window(std::exchange(other.window, VK_NULL_HANDLE))
	, surface(std::exchange(other.surface, VK_NULL_HANDLE))
	, swapchain(std::exchange(other.swapchain, VK_NULL_HANDLE))
	, swapImages(std::move(other.swapImages))
	, swapViews(std::move(other.swapViews))
	, swapchainFormat(other.swapchainFormat)
	, swapchainExtent(other.swapchainExtent) {}

VulkanWindow& VulkanWindow::operator=(VulkanWindow&& other) noexcept {
	std::swap(instance, other.instance);
	std::swap(physicalDevice, other.physicalDevice);
	std::swap(device, other.device);
	std::swap(graphicsFamilyIndex, other.graphicsFamilyIndex);
	std::swap(graphicsQueue, other.graphicsQueue);
	std::swap(presentFamilyIndex, other.presentFamilyIndex);
	std::swap(presentQueue, other.presentQueue);
	std::swap(debugMessenger, other.debugMessenger);
	std::swap(window, other.window);
	std::swap(surface, other.surface);
	std::swap(swapchain, other.swapchain);
	std::swap(swapImages, other.swapImages);
	std::swap(swapViews, other.swapViews);
	std::swap(swapchainFormat, other.swapchainFormat);
	std::swap(swapchainExtent, other.swapchainExtent);
	return *this;
}

std::unique_ptr<VulkanWindow> initialiseVulkanWindow() {
	std::unique_ptr<VulkanWindow> window = std::make_unique<VulkanWindow>();

	// Initialize Volk
	if (const auto res = volkInitialize(); VK_SUCCESS != res) {
		throw Utils::Error("Unable to load Vulkan API\n Volk returned error %s", Utils::toString(res).c_str());
	}

	// Initialize GLFW
	if (GLFW_TRUE != glfwInit()) {
		const char* errMsg = nullptr;
		glfwGetError(&errMsg);

		throw Utils::Error("GLFW initialisation failed: %s", errMsg);
	}

	if (!glfwVulkanSupported())
		throw Utils::Error("GLFW: Vulkan not supported");

	// Check for instance layers and extensions
	const auto supportedLayers = Utils::getInstanceLayers();
	const auto supportedExtensions = Utils::getInstanceExtensions();

	bool enableDebugUtils = false;

	std::vector<const char*> enabledLayers, enabledExensions;

	std::uint32_t reqExtCount = 0;
	const char** requiredExt = glfwGetRequiredInstanceExtensions(&reqExtCount);

	for (std::uint32_t i = 0; i < reqExtCount; ++i) {
		if (!supportedExtensions.count(requiredExt[i]))
			throw Utils::Error("GLFW/Vulkan: Required instance extension %s not supported", requiredExt[i]);

		enabledExensions.emplace_back(requiredExt[i]);
	}

	// Validation layers support.
#if !defined(NDEBUG) // debug builds only
	if (supportedLayers.count("VK_LAYER_KHRONOS_validation")) {
		enabledLayers.emplace_back("VK_LAYER_KHRONOS_validation");
	}

	if (supportedExtensions.count("VK_EXT_debug_utils")) {
		enableDebugUtils = true;
		enabledExensions.emplace_back("VK_EXT_debug_utils");
	}
#endif // debug builds

	for (const auto& layer : enabledLayers)
		std::fprintf(stderr, "Enabling layer: %s\n", layer);

	for (const auto& extension : enabledExensions)
		std::fprintf(stderr, "Enabling instance extension: %s\n", extension);

	// Create Vulkan instance
	window->instance = Utils::createInstance(enabledLayers, enabledExensions, enableDebugUtils);

	// Load rest of the Vulkan API
	volkLoadInstance(window->instance);

	// Setup debug messenger
	if (enableDebugUtils)
		window->debugMessenger = Utils::createDebugMessenger(window->instance);

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

	window->window = glfwCreateWindow(2048, 2048, "Vulkan Renderer", nullptr, nullptr);
	if (!window->window) {
		const char* errMsg = nullptr;
		glfwGetError(&errMsg);

		throw Utils::Error("Unable to create GLFW window\n Last error = %s", errMsg);
	}

	if (const auto res = glfwCreateWindowSurface(window->instance, window->window, nullptr, &window->surface); VK_SUCCESS != res)
		throw Utils::Error("Unable to create VkSurfaceKHR\n glfwCreateWindowSurface() returned %s", Utils::toString(res).c_str());

	// Select appropriate Vulkan device
	window->physicalDevice = selectDevice(window->instance, window->surface);
	if (VK_NULL_HANDLE == window->physicalDevice)
		throw Utils::Error("No suitable physical device found!");

	{
		VkPhysicalDeviceProperties props;
		vkGetPhysicalDeviceProperties(window->physicalDevice, &props);
		std::fprintf(stderr, "Selected device: %s (%d.%d.%d)\n", props.deviceName, VK_API_VERSION_MAJOR(props.apiVersion), VK_API_VERSION_MINOR(props.apiVersion), VK_API_VERSION_PATCH(props.apiVersion));
	}

	// Create a logical device
	// Enable required extensions. The device selection method ensures that
	// the VK_KHR_swapchain extension is present, so we can safely just
	// request it without further checks.
	std::vector<const char*> enabledDevExensions;

	enabledDevExensions.emplace_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

	for (const auto& ext : enabledDevExensions)
		std::fprintf(stderr, "Enabling device extension: %s\n", ext);

	// We need one or two queues:
	// - best case: one GRAPHICS queue that can present
	// - otherwise: one GRAPHICS queue and any queue that can present
	std::vector<std::uint32_t> queueFamilyIndices;

	if (const auto index = findQueueFamily(window->physicalDevice, VK_QUEUE_GRAPHICS_BIT, window->surface)) {
		window->graphicsFamilyIndex = *index;

		queueFamilyIndices.emplace_back(*index);
	} else {
		auto graphics = findQueueFamily(window->physicalDevice, VK_QUEUE_GRAPHICS_BIT);
		auto present = findQueueFamily(window->physicalDevice, 0, window->surface);

		assert(graphics && present);

		window->graphicsFamilyIndex = *graphics;
		window->presentFamilyIndex = *present;

		queueFamilyIndices.emplace_back(*graphics);
		queueFamilyIndices.emplace_back(*present);
	}

	window->device = createDevice(*window, window->physicalDevice, queueFamilyIndices, enabledDevExensions);

	// Retrieve VkQueues
	vkGetDeviceQueue(window->device->device, window->graphicsFamilyIndex, 0, &window->graphicsQueue);

	assert(VK_NULL_HANDLE != window->graphicsQueue);

	if (queueFamilyIndices.size() >= 2)
		vkGetDeviceQueue(window->device->device, window->presentFamilyIndex, 0, &window->presentQueue);
	else {
		window->presentFamilyIndex = window->graphicsFamilyIndex;
		window->presentQueue = window->graphicsQueue;
	}

	// Create swap chain
	std::tie(window->swapchain, window->swapchainFormat, window->swapchainExtent, window->minImageCount) = 
		createSwapchain(window->physicalDevice, window->surface, window->device->device, window->window, queueFamilyIndices);

	// Get swap chain images & create associated image views
	getSwapchainImages(window->device->device, window->swapchain, window->swapImages);
	createSwapchainImageViews(window->device->device, window->swapchainFormat, window->swapImages, window->swapViews);

	// Done
	return window;
}

VkPhysicalDevice selectDevice(VkInstance instance, VkSurfaceKHR surface) {
	std::uint32_t numDevices = 0;
	if (const auto res = vkEnumeratePhysicalDevices(instance, &numDevices, nullptr); VK_SUCCESS != res) {
		throw Utils::Error("Unable to get physical device count\n"
			"vkEnumeratePhysicalDevices() returned %s", Utils::toString(res).c_str()
		);
	}

	std::vector<VkPhysicalDevice> devices(numDevices, VK_NULL_HANDLE);
	if (const auto res = vkEnumeratePhysicalDevices(instance, &numDevices, devices.data()); VK_SUCCESS != res) {
		throw Utils::Error("Unable to get physical device list\n"
			"vkEnumeratePhysicalDevices() returned %s", Utils::toString(res).c_str()
		);
	}

	float bestScore = -1.f;
	VkPhysicalDevice bestDevice = VK_NULL_HANDLE;

	for (const auto device : devices) {
		const float score = scoreDevice(device, surface);
		if (score > bestScore) {
			bestScore = score;
			bestDevice = device;
		}
	}

	return bestDevice;
}

float scoreDevice(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) {
	VkPhysicalDeviceProperties props;
	vkGetPhysicalDeviceProperties(physicalDevice, &props);

	// Only consider Vulkan 1.1 devices
	const auto major = VK_API_VERSION_MAJOR(props.apiVersion);
	const auto minor = VK_API_VERSION_MINOR(props.apiVersion);

	if (major < 1 || (major == 1 && minor < 2)) {
		std::fprintf(stderr, "Info: Discarding device '%s': insufficient vulkan version\n", props.deviceName);
		return -1.f;
	}

	const auto exts = Utils::getDeviceExtensions(physicalDevice);

	if (!exts.count(VK_KHR_SWAPCHAIN_EXTENSION_NAME)) {
		std::fprintf(stderr, "Info: Discarding device '%s': extension %s missing\n", props.deviceName, VK_KHR_SWAPCHAIN_EXTENSION_NAME);
		return -1.0f;
	}

	if (!findQueueFamily(physicalDevice, 0, surface)) {
		std::fprintf(stderr, "Info: Discarding device '%s': can't present to surface\n", props.deviceName);
		return -1.0f;
	}

	if (!findQueueFamily(physicalDevice, VK_QUEUE_GRAPHICS_BIT)) {
		std::fprintf(stderr, "Info: Discarding device '%s': no graphics queue family\n", props.deviceName);
		return -1.0f;
	}

	// Discrete GPU > Integrated GPU > others
	float score = 0.f;

	if (VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU == props.deviceType)
		score += 500.f;
	else if (VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU == props.deviceType)
		score += 100.f;

	return score;
}

// Note: this finds *any* queue that supports the queueFlags. As such,
//   findQueueFamily( ..., VK_QUEUE_TRANSFER_BIT, ... );
// might return a GRAPHICS queue family, since GRAPHICS queues typically
// also set TRANSFER (and indeed most other operations; GRAPHICS queues are
// required to support those operations regardless). If you wanted to find
// a dedicated TRANSFER queue (e.g., such as those that exist on NVIDIA
// GPUs), you would need to use different logic.
std::optional<std::uint32_t> findQueueFamily(VkPhysicalDevice physicalDevice, VkQueueFlags queueFlags, VkSurfaceKHR surface) {
	std::uint32_t numQueues = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &numQueues, nullptr);

	std::vector<VkQueueFamilyProperties> families(numQueues);
	vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &numQueues, families.data());

	for (std::uint32_t i = 0; i < numQueues; ++i) {
		const auto& family = families[i];

		if (queueFlags == (queueFlags & family.queueFlags)) {
			if (VK_NULL_HANDLE == surface)
				return i;

			VkBool32 supported = VK_FALSE;
			const auto res = vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, surface, &supported);

			if (VK_SUCCESS == res && supported)
				return i;
		}
	}

	return {};
}

std::unique_ptr<VulkanDevice> createDevice(VulkanWindow& window, VkPhysicalDevice physicalDevice, const std::vector<std::uint32_t>& queueFamilies, const std::vector<const char*>& enabledExtensions) {
	if (queueFamilies.empty())
		throw Utils::Error("createDevice(): no queues requested");

	float queuePriorities[1] = { 1.f };

	std::vector<VkDeviceQueueCreateInfo> queueInfos(queueFamilies.size());
	for (std::size_t i = 0; i < queueFamilies.size(); ++i) {
		auto& queueInfo = queueInfos[i];
		queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queueInfo.queueFamilyIndex = queueFamilies[i];
		queueInfo.queueCount = 1;
		queueInfo.pQueuePriorities = queuePriorities;
	}

	VkPhysicalDeviceFeatures deviceFeatures{};
	vkGetPhysicalDeviceFeatures(window.physicalDevice, &deviceFeatures);

	VkPhysicalDeviceFeatures enabledFeatures{};
	if (deviceFeatures.samplerAnisotropy) {
		enabledFeatures.samplerAnisotropy = VK_TRUE;
		std::fprintf(stderr, "Enabling device feature: samplerAnisotropy\n");
	}

	window.deviceFeatures = enabledFeatures;

	VkDeviceCreateInfo deviceInfo{};
	deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

	deviceInfo.queueCreateInfoCount = std::uint32_t(queueInfos.size());
	deviceInfo.pQueueCreateInfos = queueInfos.data();

	deviceInfo.enabledExtensionCount = std::uint32_t(enabledExtensions.size());
	deviceInfo.ppEnabledExtensionNames = enabledExtensions.data();

	deviceInfo.pEnabledFeatures = &enabledFeatures;

	VkDevice device = VK_NULL_HANDLE;
	if (auto const res = vkCreateDevice(physicalDevice, &deviceInfo, nullptr, &device); VK_SUCCESS != res) {
		throw Utils::Error("Unable to create logical device\n"
			"vkCreateDevice() returned %s", Utils::toString(res).c_str()
		);
	}

	return std::make_unique<VulkanDevice>(device, window);
}

std::tuple<VkSwapchainKHR, VkFormat, VkExtent2D, std::uint32_t> createSwapchain(
	VkPhysicalDevice physicalDevice, 
	VkSurfaceKHR surface, 
	VkDevice device, 
	GLFWwindow* glfwWindow, 
	const std::vector<std::uint32_t>& queueFamilyIndices, 
	VkSwapchainKHR oldSwapchain) {
	const auto formats = getSurfaceFormats(physicalDevice, surface);
	const auto modes = getPresentModes(physicalDevice, surface);

	VkSurfaceFormatKHR format = formats[0];
	for (const auto fmt : formats) {
		if (VK_FORMAT_R8G8B8A8_SRGB == fmt.format && VK_COLOR_SPACE_SRGB_NONLINEAR_KHR == fmt.colorSpace) {
			format = fmt;
			break;
		}

		if (VK_FORMAT_B8G8R8A8_SRGB == fmt.format && VK_COLOR_SPACE_SRGB_NONLINEAR_KHR == fmt.colorSpace) {
			format = fmt;
			break;
		}
	}

	VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
	if (modes.count(VK_PRESENT_MODE_FIFO_RELAXED_KHR))
		presentMode = VK_PRESENT_MODE_FIFO_RELAXED_KHR;

	VkSurfaceCapabilitiesKHR caps;
	if (const auto res = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &caps); VK_SUCCESS != res)
		throw Utils::Error("Unable to get surface capabilities\n vkGetPhysicalDeviceSurfaceCapabilitiesKHR() returned %s", Utils::toString(res).c_str());

	std::uint32_t imageCount = 2;

	if (imageCount < caps.minImageCount + 1)
		imageCount = caps.minImageCount + 1;

	if (caps.maxImageCount > 0 && imageCount > caps.maxImageCount)
		imageCount = caps.maxImageCount;

	VkExtent2D extent = caps.currentExtent;
	if (std::numeric_limits<std::uint32_t>::max() == extent.width) {
		int width, height;
		glfwGetFramebufferSize(glfwWindow, &width, &height);

		const auto& min = caps.minImageExtent;
		const auto& max = caps.maxImageExtent;

		extent.width = std::clamp(std::uint32_t(width), min.width, max.width);
		extent.height = std::clamp(std::uint32_t(height), min.height, max.height);
	}

	VkSwapchainCreateInfoKHR chainInfo{};
	chainInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
	chainInfo.surface = surface;
	chainInfo.minImageCount = imageCount;
	chainInfo.imageFormat = format.format;
	chainInfo.imageColorSpace = format.colorSpace;
	chainInfo.imageExtent = extent;
	chainInfo.imageArrayLayers = 1;
	chainInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	chainInfo.preTransform = caps.currentTransform;
	chainInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	chainInfo.presentMode = presentMode;
	chainInfo.clipped = VK_TRUE;
	chainInfo.oldSwapchain = oldSwapchain;

	if (queueFamilyIndices.size() <= 1) {
		chainInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
	} else {
		chainInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
		chainInfo.queueFamilyIndexCount = std::uint32_t(queueFamilyIndices.size());
		chainInfo.pQueueFamilyIndices = queueFamilyIndices.data();
	}

	VkSwapchainKHR chain = VK_NULL_HANDLE;
	if (const auto res = vkCreateSwapchainKHR(device, &chainInfo, nullptr, &chain); VK_SUCCESS != res)
		throw Utils::Error("Unable to create swap chain\n vkCreateSwapchainKHR() returned %s", Utils::toString(res).c_str());

	return { chain, format.format, extent, imageCount };
}

std::vector<VkSurfaceFormatKHR> getSurfaceFormats(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) {
	std::uint32_t numFormats = 0;
	if (const auto res = vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &numFormats, nullptr); VK_SUCCESS != res)
		throw Utils::Error("Unable to get surface formats\n vkGetPhysicalDeviceSurfaceFormatsKHR() returned %s", Utils::toString(res).c_str());

	std::vector<VkSurfaceFormatKHR> formats(numFormats);
	if (const auto res = vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &numFormats, formats.data()); VK_SUCCESS != res)
		throw Utils::Error("Unable to get surface formats\n vkGetPhysicalDeviceSurfaceFormatsKHR() returned %s", Utils::toString(res).c_str());

	return formats;
}

std::unordered_set<VkPresentModeKHR> getPresentModes(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) {
	std::uint32_t numModes = 0;
	if (const auto res = vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &numModes, nullptr); VK_SUCCESS != res)
		throw Utils::Error("Unable to get present modes\n vkGetPhysicalDeviceSurfacePresentModesKHR() returned %s", Utils::toString(res).c_str());

	std::vector<VkPresentModeKHR> modes(numModes);
	if (const auto res = vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &numModes, modes.data()); VK_SUCCESS != res)
		throw Utils::Error("Unable to get present modes\n vkGetPhysicalDeviceSurfacePresentModesKHR() returned %s", Utils::toString(res).c_str());

	std::unordered_set<VkPresentModeKHR> res;
	for (const auto& mode : modes) {
		res.insert(mode);
	}

	return res;
}

void getSwapchainImages(VkDevice device, VkSwapchainKHR swapchain, std::vector<VkImage>& images) {
	assert(0 == images.size());

	std::uint32_t numImages = 0;
	if (const auto res = vkGetSwapchainImagesKHR(device, swapchain, &numImages, nullptr); VK_SUCCESS != res)
		throw Utils::Error("Unable to get swapchain images\n vkGetSwapchainImagesKHR() returned %s", Utils::toString(res).c_str());

	std::vector<VkImage> imagesTemp(numImages);
	if (const auto res = vkGetSwapchainImagesKHR(device, swapchain, &numImages, imagesTemp.data()); VK_SUCCESS != res)
		throw Utils::Error("Unable to get swapchain images\n vkGetSwapchainImagesKHR() returned %s", Utils::toString(res).c_str());

	std::swap(images, imagesTemp);
}

void createSwapchainImageViews(VkDevice device, VkFormat swapchainFormat, const std::vector<VkImage>& images, std::vector<VkImageView>& imageViews) {
	assert(0 == imageViews.size());

	for (std::size_t i = 0; i < images.size(); ++i) {
		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = images[i];
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = swapchainFormat;
		viewInfo.components = VkComponentMapping{
			VK_COMPONENT_SWIZZLE_IDENTITY,
			VK_COMPONENT_SWIZZLE_IDENTITY,
			VK_COMPONENT_SWIZZLE_IDENTITY,
			VK_COMPONENT_SWIZZLE_IDENTITY
		};
		viewInfo.subresourceRange = VkImageSubresourceRange{
			VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1
		};

		VkImageView view = VK_NULL_HANDLE;
		if (const auto res = vkCreateImageView(device, &viewInfo, nullptr, &view); VK_SUCCESS != res)
			throw Utils::Error("Unable to create image view for swap chain image %zu\n vkCreateImageView() returned %s", i, Utils::toString(res).c_str());

		imageViews.emplace_back(view);
	}

	assert(imageViews.size() == images.size());
}

SwapChanges recreateSwapchain(VulkanWindow& window) {
	const auto oldFormat = window.swapchainFormat;
	const auto oldExtent = window.swapchainExtent;

	VkSwapchainKHR oldSwapchain = window.swapchain;

	for (auto view : window.swapViews)
		vkDestroyImageView(window.device->device, view, nullptr);

	window.swapViews.clear();
	window.swapImages.clear();

	std::vector<std::uint32_t> queueFamilyIndices;
	if (window.presentFamilyIndex != window.graphicsFamilyIndex) {
		queueFamilyIndices.emplace_back(window.graphicsFamilyIndex);
		queueFamilyIndices.emplace_back(window.presentFamilyIndex);
	}

	try {
		std::tie(window.swapchain, window.swapchainFormat, window.swapchainExtent, window.minImageCount) =
			createSwapchain(window.physicalDevice, window.surface, window.device->device, window.window, queueFamilyIndices, oldSwapchain);
	} catch (...) {
		window.swapchain = oldSwapchain;
		throw;
	}

	vkDestroySwapchainKHR(window.device->device, oldSwapchain, nullptr);

	getSwapchainImages(window.device->device, window.swapchain, window.swapImages);
	createSwapchainImageViews(window.device->device, window.swapchainFormat, window.swapImages, window.swapViews);

	SwapChanges ret{};

	if (oldExtent.width != window.swapchainExtent.width || oldExtent.height != window.swapchainExtent.height)
		ret.changedSize = true;
	if (oldFormat != window.swapchainFormat)
		ret.changedFormat = true;

	return ret;
}

VkResult submitAndPresent(
	const VulkanWindow& window, 
	std::vector<VkCommandBuffer>& cmdBuffers, 
	std::vector<vk::Fence>& frameDone, 
	std::vector<vk::Semaphore>& imageAvailable, 
	std::vector<vk::Semaphore>& renderFinished, 
	std::size_t frameIndex, 
	std::uint32_t imageIndex) 
{
	// Submit
	VkPipelineStageFlags waitPipelineStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

	VkSubmitInfo subInfo{};
	subInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	subInfo.commandBufferCount = 1;
	subInfo.pCommandBuffers = &cmdBuffers[frameIndex];
	subInfo.waitSemaphoreCount = 1;
	subInfo.pWaitSemaphores = &imageAvailable[frameIndex].handle;
	subInfo.pWaitDstStageMask = &waitPipelineStages;
	subInfo.signalSemaphoreCount = 1;
	subInfo.pSignalSemaphores = &renderFinished[frameIndex].handle;

	if (const auto res = vkQueueSubmit(window.graphicsQueue, 1, &subInfo, frameDone[frameIndex].handle); VK_SUCCESS != res)
		throw Utils::Error("Unable to submit command buffer to queue\n vkQueueSubmit() returned %s", Utils::toString(res).c_str());

	// Present
	VkPresentInfoKHR presentInfo{};
	presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	presentInfo.waitSemaphoreCount = 1;
	presentInfo.pWaitSemaphores = &renderFinished[frameIndex].handle;
	presentInfo.swapchainCount = 1;
	presentInfo.pSwapchains = &window.swapchain;
	presentInfo.pImageIndices = &imageIndex;
	presentInfo.pResults = nullptr;

	return vkQueuePresentKHR(window.presentQueue, &presentInfo);
}