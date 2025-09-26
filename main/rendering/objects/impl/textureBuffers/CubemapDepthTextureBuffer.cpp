#include "CubemapDepthTextureBuffer.hpp"

#include "../../../../vulkan/VulkanContext.hpp"
#include "../../../PipelineCreation.hpp"

#include "../../../../vulkan/VulkanDevice.hpp"

#include "Error.hpp"
#include "toString.hpp"

CubemapDepthTextureBuffer::CubemapDepthTextureBuffer(
	VulkanContext* context,
	VkFormat format,
	VkSampleCountFlagBits* sampleCount,
	VkExtent2D* renderExtent
) : TextureBuffer(context) {
	this->format = format;
	this->sampleCount = sampleCount;

	if (!renderExtent)
		this->renderExtent = &this->context->window->swapchainExtent;
	else
		this->renderExtent = renderExtent;

	this->recreate();
}

void CubemapDepthTextureBuffer::recreate() {
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
	imageInfo.samples = this->sampleCount ? *this->sampleCount : VK_SAMPLE_COUNT_1_BIT;
	imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
	imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

	VmaAllocationCreateInfo allocInfo{};
	allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

	VkImage image = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;

	if (const auto res = vmaCreateImage(this->context->allocator->allocator, &imageInfo, &allocInfo, &image, &allocation, nullptr); VK_SUCCESS != res) {
		std::fprintf(stderr, "Unable to allocate depth buffer image.\n vmaCreateImage() returned %s\n", Utils::toString(res).c_str());
		throw Utils::Error("Unable to allocate depth buffer image.\n vmaCreateImage() returned %s\n", Utils::toString(res).c_str());
	}

	vk::Image Image(this->context->allocator->allocator, image, allocation);
	this->image = std::move(Image);

	// Create the image view info initially with CUBE view type and 6 layerCount for the samplerCubeShadow descriptor
	VkImageViewCreateInfo viewInfo{};
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.image = this->image.image;
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
	viewInfo.format = this->format;
	viewInfo.components = VkComponentMapping{};
	viewInfo.subresourceRange = VkImageSubresourceRange{ VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 6 };

	VkImageView view = VK_NULL_HANDLE;
	if (const auto res = vkCreateImageView(this->context->window->device->device, &viewInfo, nullptr, &view); VK_SUCCESS != res)
		throw Utils::Error("Unable to create image view.\n vkCreateImageView() returned %s", Utils::toString(res).c_str());

	this->descriptorView = vk::ImageView(this->context->window->device->device, view);

	// Alter image view info with 2D view type and layerCount 1 for framebuffer views
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	viewInfo.subresourceRange.layerCount = 1;

	this->framebufferViews.clear();
	for (std::uint32_t i = 0; i < 6; i++) {
		viewInfo.subresourceRange.baseArrayLayer = i;

		VkImageView framebufferView = VK_NULL_HANDLE;
		if (const auto res = vkCreateImageView(this->context->window->device->device, &viewInfo, nullptr, &framebufferView); VK_SUCCESS != res)
			throw Utils::Error("Unable to create image view.\n vkCreateImageView() returned %s", Utils::toString(res).c_str());

		this->framebufferViews.emplace_back(vk::ImageView(this->context->window->device->device, framebufferView));
	}

	TextureBuffer::recreate();
}

vk::ImageView& CubemapDepthTextureBuffer::getImageView() {
	return this->descriptorView;
}

// Downcasting will be required to access this method!
std::vector<vk::ImageView>& CubemapDepthTextureBuffer::getFramebufferViews() {
	return this->framebufferViews;
}
