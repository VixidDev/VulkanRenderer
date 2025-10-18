#pragma once

#include <vector>
#include <map>
#include <string>

#include <volk/volk.h>

class VulkanWindow;

static std::map<VkPresentModeKHR, std::string> presentModesMapStrings{
	{ VK_PRESENT_MODE_IMMEDIATE_KHR, "Immediate" },
	{ VK_PRESENT_MODE_MAILBOX_KHR, "Mailbox" },
	{ VK_PRESENT_MODE_FIFO_KHR, "FIFO" },
	{ VK_PRESENT_MODE_FIFO_RELAXED_KHR, "FIFO Relaxed" }
};

struct SwapChanges {
	bool changedSize : 1;
	bool changedFormat : 1;
};

class Swapchain {
public:
	Swapchain() = default;
	Swapchain(VulkanWindow* window);
	~Swapchain();

	Swapchain(const Swapchain&) = delete;
	Swapchain& operator=(const Swapchain&) = delete;
	Swapchain(Swapchain&&) = delete;
	Swapchain& operator=(Swapchain&&) = delete;

	SwapChanges recreate(bool firstTime = false);

	VkSwapchainKHR& getHandle();

	VkFormat getFormat();
	int& getPresentMode();
	const std::vector<std::string>& getPresentModeStrings() const;

	std::uint32_t getMinImageCount();
	VkExtent2D& getExtent();
	VkExtent2D& getHalfExtent();

	const std::vector<VkImageView>& getViews() const;
	VkImage getImage(std::uint32_t imageIndex);

	static const int MAX_FRAMES_IN_FLIGHT = 3;
private:
	std::vector<VkSurfaceFormatKHR> getSurfaceFormats();
	void getPresentModes();

	VkSurfaceFormatKHR determineFormat(const std::vector<VkSurfaceFormatKHR>& formats);

	void getSwapchainImages();
	void createSwapchainImageViews();

	VulkanWindow* window = nullptr;

	VkSwapchainKHR swapchain = VK_NULL_HANDLE;

	VkSurfaceFormatKHR selectedFormat;
	std::vector<VkPresentModeKHR> presentModes;
	int presentMode = 0;

	std::uint32_t minImageCount = 0;
	VkExtent2D swapchainExtent;
	VkExtent2D halfSwapchainExtent;

	std::vector<VkImage> swapchainImages;
	std::vector<VkImageView> swapchainViews;

	VkSurfaceCapabilitiesKHR surfaceCaps;
	std::vector<std::string> presentModeStrings;
};