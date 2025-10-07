#include "VulkanWindow.hpp"

#include <chrono>

#include "ContextHelpers.hpp"
#include "VulkanDevice.hpp"
#include "Swapchain.hpp"
#include "Error.hpp"
#include "toString.hpp"

#if !defined(GLFW_INCLUDE_NONE)
#	define GLFW_INCLUDE_NONE 1
#endif
#include <GLFW/glfw3.h>

VulkanWindow::~VulkanWindow() {
	// Destory swapchain first
	this->swapchain.reset();

	// Window and related objects
	if (this->surface != VK_NULL_HANDLE)
		vkDestroySurfaceKHR(this->instance, this->surface, nullptr);

	if (this->glfwWindow) {
		glfwDestroyWindow(this->glfwWindow);

		// The following assumes that we never create more than one window;
		// if there are multiple windows, destroying one of them would
		// unload the whole GLFW library. Nevertheless, this solution is
		// convenient when only dealing with one window as it ensures that 
		// GLFW is unloaded after all window-related resources are.
		glfwTerminate();
	}

	// Destroy device
	this->device.reset();

	// Destory instance related objects
	if (debugMessenger != VK_NULL_HANDLE)
		vkDestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);

	if (instance != VK_NULL_HANDLE)
		vkDestroyInstance(instance, nullptr);
}

VulkanWindow::VulkanWindow(VulkanWindow&& other) noexcept
	: instance(std::exchange(other.instance, VK_NULL_HANDLE))
	, debugMessenger(std::exchange(other.debugMessenger, VK_NULL_HANDLE))
	, glfwWindow(std::exchange(other.glfwWindow, nullptr))
	, surface(std::exchange(other.surface, VK_NULL_HANDLE))
	, device(std::exchange(other.device, nullptr))
	, swapchain(std::exchange(other.swapchain, nullptr)) {}

VulkanWindow& VulkanWindow::operator=(VulkanWindow&& other) noexcept {
	std::swap(instance, other.instance);
	std::swap(debugMessenger, other.debugMessenger);
	std::swap(glfwWindow, other.glfwWindow);
	std::swap(surface, other.surface);
	std::swap(device, other.device);
	std::swap(swapchain, other.swapchain);
	return *this;
}


VulkanWindow::VulkanWindow() {
	// Initialise volk
	if (const VkResult res = volkInitialize(); VK_SUCCESS != res)
		throw Utils::Error("Volk: Unable to load Vulkan API\nReturned error %s", Utils::toString(res).c_str());

	// Initialise GLFW
	if (glfwInit() != GLFW_TRUE) {
		const char* errMsg = nullptr;
		glfwGetError(&errMsg);
		throw Utils::Error("GLFW: Initialisation failed: %s", errMsg);
	}

	// Check Vulkan is supported by GLFW
	if (!glfwVulkanSupported())
		throw Utils::Error("GLFW: Vulkan not supported");

	std::vector<std::string> requestedLayers = {
#ifndef NDEBUG
		"VK_LAYER_KRHONOS_validation",
#endif
	};
	std::vector<std::string> requestedExtensions = {
#ifndef NDEBUG
		"VK_EXT_debug_utils",
#endif
	};

	// Enable layers and extensions
	this->enableLayers(requestedLayers);
	this->enableExtensions(requestedExtensions);

	// Create Vulkan instance
	this->createInstance();

	// Load rest of Vulkan API
	volkLoadInstance(this->instance);

#ifndef NDEBUG
	// Setup persistent debug messenger
	this->createDebugMessenger();
#endif

	// Create GLFW window and Vulkan surface
	this->createGLFWwindow();
	this->createSurface();

	// Create VulkanDevice
	this->device = std::make_unique<VulkanDevice>(this->instance, this->surface);

	this->swapchain = std::make_unique<Swapchain>(this);
}

void VulkanWindow::enableLayers(std::vector<std::string>& requestedLayers) {
	// Get supported layers
	const std::unordered_set<std::string> supportedLayers = Utils::getInstanceLayers();

	for (const std::string& requestedLayer : requestedLayers) {
		if (supportedLayers.contains(requestedLayer)) {
			this->enabledLayers.emplace_back(requestedLayer.c_str());
			std::fprintf(stderr, "Enabling layer: %s\n", requestedLayer.c_str());
		}
	}
}

void VulkanWindow::enableExtensions(std::vector<std::string>& requestedExtensions) {
	// Get supported extensions
	const std::unordered_set<std::string> supportedExtensions = Utils::getInstanceExtensions();

	// Get any required extensions from GLFW
	std::uint32_t reqExtCount = 0;
	const char** requiredExtensions = glfwGetRequiredInstanceExtensions(&reqExtCount);

	// Check we support the required extensions, and add them if so
	for (std::uint32_t i = 0; i < reqExtCount; i++) {
		if (!supportedExtensions.contains(requiredExtensions[i]))
			throw Utils::Error("GLFW/Vulkan: Required instance extension %s not supported", requiredExtensions[i]);
	
		this->enabledExtensions.emplace_back(requiredExtensions[i]);
		std::fprintf(stderr, "Enabling extension: %s\n", requiredExtensions[i]);
	}

	// Check we support the requested extensions, and add if so
	for (const std::string& requestedExtension : requestedExtensions) {
		if (supportedExtensions.contains(requestedExtension)) {
			this->enabledExtensions.emplace_back(requestedExtension.c_str());
			std::fprintf(stderr, "Enabling extension: %s\n", requestedExtension.c_str());
		}
	}
}

void VulkanWindow::createInstance() {
	// Prepare debug messenger for instance creation and destruction
	VkDebugUtilsMessengerCreateInfoEXT debugInfo{};
#ifndef NDEBUG
	debugInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
	debugInfo.messageSeverity = 
		// VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | 
		// VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
		VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | 
		VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
	debugInfo.messageType = 
		VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | 
		VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | 
		VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
	debugInfo.pfnUserCallback = &Utils::debugUtilCallback;
	debugInfo.pUserData = nullptr;
#endif

	// Prepare application info
	// The `apiVersion` is the *highest* version of Vulkan than the
	// application can use. We can therefore safely set it to 1.3, even if
	// we are not intending to use any 1.4 features (and want to run on
	// pre-1.3 implementations).
	VkApplicationInfo appInfo{};
	appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	appInfo.pApplicationName = "VulkanRenderer";
	appInfo.applicationVersion = 1;
	appInfo.apiVersion = VK_MAKE_API_VERSION(0, 1, 3, 0); // Version 1.3

	// Create instance
	VkInstanceCreateInfo instanceInfo{};
	instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	instanceInfo.enabledLayerCount = std::uint32_t(this->enabledLayers.size());
	instanceInfo.ppEnabledLayerNames = this->enabledLayers.data();
	instanceInfo.enabledExtensionCount = std::uint32_t(this->enabledExtensions.size());
	instanceInfo.ppEnabledExtensionNames = this->enabledExtensions.data();
	instanceInfo.pApplicationInfo = &appInfo;

#ifndef NDEBUG
	debugInfo.pNext = instanceInfo.pNext;
	instanceInfo.pNext = &debugInfo;
#endif

	if (const VkResult res = vkCreateInstance(&instanceInfo, nullptr, &this->instance); VK_SUCCESS != res)
		throw Utils::Error("Unable to create Vulkan instance\nvkCreateInstance() returned %s", Utils::toString(res).c_str());
}

void VulkanWindow::createDebugMessenger() {
	// Set up the debug messaging for the rest of the application
	VkDebugUtilsMessengerCreateInfoEXT debugInfo{};
	debugInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
	debugInfo.messageSeverity = 
		// VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | 
		// VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT | 
		VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | 
		VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
	debugInfo.messageType = 
		VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | 
		VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | 
		VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
	debugInfo.pfnUserCallback = &Utils::debugUtilCallback;
	debugInfo.pUserData = nullptr;

	if (const auto res = vkCreateDebugUtilsMessengerEXT(instance, &debugInfo, nullptr, &this->debugMessenger); VK_SUCCESS != res)
		throw Utils::Error("Unable to set up debug messenger\nvkCreateDebugUtilsMessengerEXT() returned %s", Utils::toString(res).c_str());
}

void VulkanWindow::createGLFWwindow() {
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

	this->glfwWindow = glfwCreateWindow(2560, 1440, "Vulkan Renderer", nullptr, nullptr);
	if (!this->glfwWindow) {
		const char* errMsg = nullptr;
		glfwGetError(&errMsg);
		throw Utils::Error("GLFW: Unable to create GLFW window\nLast error = %s", errMsg);
	}
}

void VulkanWindow::createSurface() {
	if (const VkResult res = glfwCreateWindowSurface(this->instance, this->glfwWindow, nullptr, &this->surface); VK_SUCCESS != res)
		throw Utils::Error("Unable to create VkSurfaceKHR\nglfwCreateWindowSurface() returned %s", Utils::toString(res).c_str());
}

VkInstance VulkanWindow::getInstance() {
	return this->instance;
}

GLFWwindow* VulkanWindow::getGLFWwindow() {
	return this->glfwWindow;
}

VkSurfaceKHR VulkanWindow::getSurface() {
	return this->surface;
}

VulkanDevice* VulkanWindow::getDevice() const {
	return this->device.get();
}

Swapchain* VulkanWindow::getSwapchain() const {
	return this->swapchain.get();
}

//std::chrono::steady_clock::time_point after = std::chrono::high_resolution_clock::now();

VkResult submitAndPresent(
	VulkanWindow& window, 
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

	if (const auto res = vkQueueSubmit(window.getDevice()->getGraphicsQueue(), 1, &subInfo, frameDone[frameIndex].handle); VK_SUCCESS != res)
		throw Utils::Error("Unable to submit command buffer to queue\n vkQueueSubmit() returned %s", Utils::toString(res).c_str());

	// Present
	VkPresentInfoKHR presentInfo{};
	presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	presentInfo.waitSemaphoreCount = 1;
	presentInfo.pWaitSemaphores = &renderFinished[frameIndex].handle;
	presentInfo.swapchainCount = 1;
	presentInfo.pSwapchains = &window.getSwapchain()->getHandle();
	presentInfo.pImageIndices = &imageIndex;
	presentInfo.pResults = nullptr;

	//auto before = std::chrono::high_resolution_clock::now();

	//auto entireFrame = std::chrono::duration_cast<std::chrono::duration<float, std::ratio<1>>>(before - after).count() * 1000.0f;
	//std::fprintf(stderr, "rest of frame took: %.4f ms\n", entireFrame);

	VkResult res = vkQueuePresentKHR(window.getDevice()->getPresentQueue(), &presentInfo);
	//after = std::chrono::high_resolution_clock::now();
	//auto difference = std::chrono::duration_cast<std::chrono::duration<float, std::ratio<1>>>(after - before).count() * 1000.0f;
	//std::fprintf(stderr, "vkQueuePresentKHR took: %.4f ms\n", difference);

	return res;
}