#include "CubemapTextureBuffer.hpp"

#include <array>

#include "../../../PipelineCreation.hpp"
#include "../../../../vulkan/VulkanContext.hpp"
#include "../../../../vulkan/VulkanDevice.hpp"
#include "../../../../vulkan/Swapchain.hpp"

#include "Error.hpp"
#include "toString.hpp"

CubemapTextureBuffer::CubemapTextureBuffer(
	VulkanContext* context,
	VkFormat format,
	VkExtent2D* renderExtent,
	bool skipIndividualImageViews,
	bool exemptFromRecreation
) : TextureBuffer(context), 
	skipIndividualImageViews(skipIndividualImageViews)
{
	this->format = format;

	if (!renderExtent)
		this->renderExtent = &this->context->window->getSwapchain()->getExtent();
	else
		this->renderExtent = renderExtent;

	std::array<VkFormat, 6> depthFormats = {
		VK_FORMAT_D16_UNORM,
		VK_FORMAT_D16_UNORM_S8_UINT,
		VK_FORMAT_X8_D24_UNORM_PACK32,
		VK_FORMAT_D24_UNORM_S8_UINT,
		VK_FORMAT_D32_SFLOAT,
		VK_FORMAT_D32_SFLOAT_S8_UINT
	};

	for (VkFormat format : depthFormats) {
		if (this->format == format) {
			this->aspectFlags = VK_IMAGE_ASPECT_DEPTH_BIT;
			this->usageFlags |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
		} else {
			this->usageFlags |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
		}
	}

	std::array<VkFormat, 4> stencilFormats = {
		VK_FORMAT_S8_UINT,
		VK_FORMAT_D16_UNORM_S8_UINT,
		VK_FORMAT_D24_UNORM_S8_UINT,
		VK_FORMAT_D32_SFLOAT_S8_UINT
	};

	for (VkFormat format : stencilFormats) {
		if (this->format == format) {
			this->aspectFlags |= VK_IMAGE_ASPECT_STENCIL_BIT;
			this->usageFlags |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
		}
	}

	this->recreate();

	this->exemptFromRecreation = exemptFromRecreation;
}

void CubemapTextureBuffer::recreate() {
	if (this->exemptFromRecreation) return;

	// Since cube maps require a different setup to usual Image-ImageView texture creation
	// I manually create the image and image views in this function

	// Create VkImage
	VkImageCreateInfo imageInfo{};
	imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageInfo.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
	imageInfo.imageType = VK_IMAGE_TYPE_2D;
	imageInfo.format = this->format;
	imageInfo.extent.width = this->renderExtent->width;
	imageInfo.extent.height = this->renderExtent->height;
	imageInfo.extent.depth = 1;
	imageInfo.mipLevels = 1;
	imageInfo.arrayLayers = 6;
	imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageInfo.usage = this->usageFlags;
	imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

	VmaAllocationCreateInfo allocInfo{};
	allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

	VkImage image = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;

	if (const VkResult res = vmaCreateImage(this->context->allocator->allocator, &imageInfo, &allocInfo, &image, &allocation, nullptr); VK_SUCCESS != res) {
		throw Utils::Error("Unable to allocate depth buffer image.\nvmaCreateImage() returned %s\n", Utils::toString(res).c_str());
	}

	vk::Image Image(this->context->allocator->allocator, image, allocation);
	this->image = std::move(Image);

	// Create the image view info initially with CUBE view type and 6 layerCount for the cube descriptor
	VkImageViewCreateInfo viewInfo{};
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.image = this->image.image;
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
	viewInfo.format = this->format;
	viewInfo.components = VkComponentMapping{};
	viewInfo.subresourceRange = VkImageSubresourceRange{ this->aspectFlags, 0, 1, 0, 6 };

	VkImageView view = VK_NULL_HANDLE;
	if (const VkResult res = vkCreateImageView(this->context->window->getDevice()->getDevice(), &viewInfo, nullptr, &view); VK_SUCCESS != res)
		throw Utils::Error("Unable to create image view.\nvkCreateImageView() returned %s", Utils::toString(res).c_str());

	this->descriptorView = vk::ImageView(this->context->window->getDevice()->getDevice(), view);

	if (this->skipIndividualImageViews) {
		TextureBuffer::recreate();
		return;
	}

	// Alter image view info with 2D view type and layerCount 1 for framebuffer views
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	viewInfo.subresourceRange.layerCount = 1;

	this->framebufferViews.clear();
	for (std::uint32_t i = 0; i < 6; i++) {
		viewInfo.subresourceRange.baseArrayLayer = i;

		VkImageView framebufferView = VK_NULL_HANDLE;
		if (const VkResult res = vkCreateImageView(this->context->window->getDevice()->getDevice(), &viewInfo, nullptr, &framebufferView); VK_SUCCESS != res)
			throw Utils::Error("Unable to create image view.\nvkCreateImageView() returned %s", Utils::toString(res).c_str());

		this->framebufferViews.emplace_back(vk::ImageView(this->context->window->getDevice()->getDevice(), framebufferView));
	}

	TextureBuffer::recreate();
}

vk::ImageView& CubemapTextureBuffer::getImageView() {
	return this->descriptorView;
}

// Downcasting will be required to access this method!
std::vector<vk::ImageView>& CubemapTextureBuffer::getFramebufferViews() {
	if (this->framebufferViews.empty()) {
		throw Utils::Error("CubemapTextureBuffer: framebufferViews is empty! Is this texture buffer to be written to?");
	}

	return this->framebufferViews;
}
