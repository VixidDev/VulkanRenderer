#include "Swapchain.hpp"

#include "VulkanWindow.hpp"
#include "VulkanDevice.hpp"
#include "Error.hpp"
#include "toString.hpp"

#include <algorithm>
#include <GLFW/glfw3.h>

Swapchain::Swapchain(VulkanWindow* window) : window(window) {
	// Get supported surface formats and present modes
	const std::vector<VkSurfaceFormatKHR> supportedFormats = this->getSurfaceFormats();
	this->getPresentModes();

	// Populate present mode strings based on supported present modes
	for (const VkPresentModeKHR& mode : this->presentModes) {
		this->presentModeStrings.emplace_back(presentModesMapStrings[mode]);
	}

	this->selectedFormat = this->determineFormat(supportedFormats);

	for (std::size_t i = 0; i < this->presentModes.size(); i++) {
		VkPresentModeKHR mode = this->presentModes.at(i);
		
		if (mode == VK_PRESENT_MODE_FIFO_KHR)
			this->presentMode = i;

		if (mode == VK_PRESENT_MODE_FIFO_RELAXED_KHR) {
			this->presentMode = i;
			break;
		}
	}

	this->recreate(true);
}

Swapchain::~Swapchain() {
	VkDevice device = this->window->getDevice()->getDevice();

	for (const VkImageView view : this->swapchainViews)
		vkDestroyImageView(device, view, nullptr);
	
	if (this->swapchain != VK_NULL_HANDLE)
		vkDestroySwapchainKHR(device, this->swapchain, nullptr);
}

SwapChanges Swapchain::recreate(bool firstTime) {
	VkPhysicalDevice physicalDevice = this->window->getDevice()->getPhysicalDevice();
	VkDevice device = this->window->getDevice()->getDevice();
	VkSurfaceKHR surface = this->window->getSurface();

	const VkSurfaceFormatKHR oldFormat = this->selectedFormat;
	const VkExtent2D		 oldExtent = this->swapchainExtent;

	VkSwapchainKHR oldSwapchain = this->swapchain;

	if (!firstTime) {
		for (VkImageView view : this->swapchainViews)
			vkDestroyImageView(device, view, nullptr);
	}

	// Get surface caps
	if (const VkResult res = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &this->surfaceCaps); VK_SUCCESS != res)
		throw Utils::Error("Unable to get surface capabilities\n vkGetPhysicalDeviceSurfaceCapabilitiesKHR() returned %s", Utils::toString(res).c_str());

	this->minImageCount = this->surfaceCaps.minImageCount + 1;
	if (this->surfaceCaps.maxImageCount > 0 && minImageCount > this->surfaceCaps.maxImageCount)
		this->minImageCount = this->surfaceCaps.maxImageCount;

	this->swapchainExtent = this->surfaceCaps.currentExtent;
	if (this->swapchainExtent.width == std::numeric_limits<std::uint32_t>::max()) {
		int width, height;
		glfwGetFramebufferSize(this->window->getGLFWwindow(), &width, &height);

		const VkExtent2D& min = this->surfaceCaps.minImageExtent;
		const VkExtent2D& max = this->surfaceCaps.maxImageExtent;

		this->swapchainExtent.width = std::clamp(std::uint32_t(width), min.width, max.width);
		this->swapchainExtent.height = std::clamp(std::uint32_t(height), min.height, max.height);
	}

	VkSwapchainCreateInfoKHR swapchainInfo{};
	swapchainInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
	swapchainInfo.surface = this->window->getSurface();
	swapchainInfo.minImageCount = this->minImageCount;
	swapchainInfo.imageFormat = this->selectedFormat.format;
	swapchainInfo.imageColorSpace = this->selectedFormat.colorSpace;
	swapchainInfo.imageExtent = this->swapchainExtent;
	swapchainInfo.imageArrayLayers = 1;
	swapchainInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	swapchainInfo.preTransform = this->surfaceCaps.currentTransform;
	swapchainInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	swapchainInfo.presentMode = this->presentModes[this->presentMode];
	swapchainInfo.clipped = VK_TRUE;
	swapchainInfo.oldSwapchain = oldSwapchain;

	const std::vector<std::uint32_t>& queueFamilyIndices = this->window->getDevice()->getQueueFamilyIndices();
	if (queueFamilyIndices.size() <= 1) {
		swapchainInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
	} else {
		swapchainInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
		swapchainInfo.queueFamilyIndexCount = std::uint32_t(queueFamilyIndices.size());
		swapchainInfo.pQueueFamilyIndices = queueFamilyIndices.data();
	}

	if (const auto res = vkCreateSwapchainKHR(this->window->getDevice()->getDevice(), &swapchainInfo, nullptr, &this->swapchain); VK_SUCCESS != res)
		throw Utils::Error("Unable to create swap chain\nvkCreateSwapchainKHR() returned %s", Utils::toString(res).c_str());

	this->getSwapchainImages();
	this->createSwapchainImageViews();

	SwapChanges ret{};

	if (oldExtent.width != this->swapchainExtent.width || oldExtent.height != this->swapchainExtent.height)
		ret.changedSize = true;
	if (oldFormat.format != this->selectedFormat.format)
		ret.changedFormat = true;

	return ret;
}

VkSwapchainKHR& Swapchain::getHandle() {
	return this->swapchain;
}

VkFormat Swapchain::getFormat() {
	return this->selectedFormat.format;
}

int& Swapchain::getPresentMode() {
	return this->presentMode;
}

const std::vector<std::string>& Swapchain::getPresentModeStrings() const {
	return this->presentModeStrings;
}

std::uint32_t Swapchain::getMinImageCount() {
	return this->minImageCount;
}

VkExtent2D& Swapchain::getExtent() {
	return this->swapchainExtent;
}

const std::vector<VkImageView>& Swapchain::getViews() const {
	return this->swapchainViews;
}

VkImage Swapchain::getImage(std::uint32_t imageIndex) {
	return this->swapchainImages[imageIndex];
}

std::vector<VkSurfaceFormatKHR> Swapchain::getSurfaceFormats() {
	VkPhysicalDevice physicalDevice = this->window->getDevice()->getPhysicalDevice();
	VkSurfaceKHR surface = this->window->getSurface();

	std::uint32_t numFormats = 0;
	if (const VkResult res = vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &numFormats, nullptr); VK_SUCCESS != res)
		throw Utils::Error("Unable to get surface formats\n vkGetPhysicalDeviceSurfaceFormatsKHR() returned %s", Utils::toString(res).c_str());

	std::vector<VkSurfaceFormatKHR> formats(numFormats);
	if (const VkResult res = vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &numFormats, formats.data()); VK_SUCCESS != res)
		throw Utils::Error("Unable to get surface formats\n vkGetPhysicalDeviceSurfaceFormatsKHR() returned %s", Utils::toString(res).c_str());

	return formats;
}

void Swapchain::getPresentModes() {
	VkPhysicalDevice physicalDevice = this->window->getDevice()->getPhysicalDevice();
	VkSurfaceKHR surface = this->window->getSurface();

	std::uint32_t numModes = 0;
	if (const VkResult res = vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &numModes, nullptr); VK_SUCCESS != res)
		throw Utils::Error("Unable to get present modes\n vkGetPhysicalDeviceSurfacePresentModesKHR() returned %s", Utils::toString(res).c_str());

	this->presentModes.resize(numModes);
	if (const VkResult res = vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &numModes, this->presentModes.data()); VK_SUCCESS != res)
		throw Utils::Error("Unable to get present modes\n vkGetPhysicalDeviceSurfacePresentModesKHR() returned %s", Utils::toString(res).c_str());
}

VkSurfaceFormatKHR Swapchain::determineFormat(const std::vector<VkSurfaceFormatKHR>& formats) {
	assert(!formats.empty() && "Supplied list of formats must contain at least 1 format!");
	
	VkSurfaceFormatKHR format = formats[0];

	for (const VkSurfaceFormatKHR& fmt : formats) {
		if (fmt.format == VK_FORMAT_R8G8B8A8_SRGB && fmt.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
			format = fmt;
			break;
		}
		if (fmt.format == VK_FORMAT_B8G8R8A8_SRGB && fmt.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
			format = fmt;
			break;
		}
	}

	return format;
}

void Swapchain::getSwapchainImages() {
	VkDevice device = this->window->getDevice()->getDevice();

	std::uint32_t numImages = 0;
	if (const VkResult res = vkGetSwapchainImagesKHR(device, this->swapchain, &numImages, nullptr); VK_SUCCESS != res)
		throw Utils::Error("Unable to get swapchain images\nvkGetSwapchainImagesKHR() returned %s", Utils::toString(res).c_str());

	this->swapchainImages.clear();
	this->swapchainImages.resize(numImages, VK_NULL_HANDLE);
	if (const VkResult res = vkGetSwapchainImagesKHR(device, this->swapchain, &numImages, this->swapchainImages.data()); VK_SUCCESS != res)
		throw Utils::Error("Unable to get swapchain images\nvkGetSwapchainImagesKHR() returned %s", Utils::toString(res).c_str());
}

void Swapchain::createSwapchainImageViews() {
	this->swapchainViews.clear();

	for (std::size_t i = 0; i < this->swapchainImages.size(); i++) {
		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = this->swapchainImages[i];
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = this->selectedFormat.format;
		viewInfo.components = VkComponentMapping{
			VK_COMPONENT_SWIZZLE_IDENTITY,
			VK_COMPONENT_SWIZZLE_IDENTITY,
			VK_COMPONENT_SWIZZLE_IDENTITY,
			VK_COMPONENT_SWIZZLE_IDENTITY
		};
		viewInfo.subresourceRange = VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

		VkImageView view = VK_NULL_HANDLE;
		if (const VkResult res = vkCreateImageView(this->window->getDevice()->getDevice(), &viewInfo, nullptr, &view); VK_SUCCESS != res)
			throw Utils::Error("Unable to create image view for swap chain image %zu\nvkCreateImageView() returned %s", i, Utils::toString(res).c_str());
	
		this->swapchainViews.emplace_back(view);
	}
}
