#pragma once

#include <vector>
#include <string>
#include <memory>

#include "objects/VkObjects.hpp"

struct GLFWwindow;
class VulkanDevice;
class Swapchain;

class VulkanWindow {
public:
	VulkanWindow();
	~VulkanWindow();

	// Move-only
	VulkanWindow(VulkanWindow const&) = delete;
	VulkanWindow& operator= (VulkanWindow const&) = delete;
	VulkanWindow(VulkanWindow&&) noexcept;
	VulkanWindow& operator= (VulkanWindow&&) noexcept;

	VkInstance getInstance();
	GLFWwindow* getGLFWwindow();
	VkSurfaceKHR getSurface();

	VulkanDevice* getDevice() const;
	Swapchain* getSwapchain() const;
private:
	void enableLayers(std::vector<std::string>& requestedLayers);
	void enableExtensions(std::vector<std::string>& requestedExtensions);

	void createInstance();
	void createDebugMessenger();
	void createGLFWwindow();
	void createSurface();

	std::vector<const char*> enabledLayers, enabledExtensions;

	VkInstance instance = VK_NULL_HANDLE;
	VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
	GLFWwindow* glfwWindow = nullptr;
	VkSurfaceKHR surface = VK_NULL_HANDLE;

	std::unique_ptr<VulkanDevice> device;
	std::unique_ptr<Swapchain> swapchain;
};

VkResult submitAndPresent(
	VulkanWindow& window,
	std::vector<VkCommandBuffer>& cmdBuffers,
	std::vector<vk::Fence>& frameDone,
	std::vector<vk::Semaphore>& imageAvailable,
	std::vector<vk::Semaphore>& renderFinished,
	std::size_t frameIndex,
	std::uint32_t imageIndex
);