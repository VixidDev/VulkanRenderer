#include "CubemapArrayDepthTextureBuffer.hpp"

#include "ShadowDepthTextureBuffer.hpp"

#include "../../../../vulkan/VulkanContext.hpp"
#include "../../../PipelineCreation.hpp"

#include "../../../../vulkan/VulkanDevice.hpp"

#include "Error.hpp"
#include "toString.hpp"

CubemapArrayDepthTextureBuffer::CubemapArrayDepthTextureBuffer(
	VulkanContext* context,
	std::uint32_t arraySize,
	VkExtent2D* renderExtent) : arraySize(arraySize), TextureBuffer(context) 
{
	if (!renderExtent)
		this->renderExtent = &this->context->window->swapchainExtent;
	else
		this->renderExtent = renderExtent;

	this->recreate();
}

void CubemapArrayDepthTextureBuffer::recreate() {
	// Create VkImage
	VkImageCreateInfo imageInfo{};
	imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageInfo.imageType = VK_IMAGE_TYPE_2D;
	imageInfo.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
	imageInfo.format = VK_FORMAT_D32_SFLOAT;
	imageInfo.extent.width = this->renderExtent->width;
	imageInfo.extent.height = this->renderExtent->height;
	imageInfo.extent.depth = 1;
	imageInfo.mipLevels = 1;
	imageInfo.arrayLayers = 6 * this->arraySize;
	imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
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

	this->image = vk::Image(this->context->allocator->allocator, image, allocation);

	// Create the image view info initially with CUBE_ARRAY view type and 6 * arraySize layerCount for the samplerCubeArray descriptor
	VkImageViewCreateInfo viewInfo{};
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.image = this->image.image;
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
	viewInfo.format = VK_FORMAT_D32_SFLOAT;
	viewInfo.components = VkComponentMapping{};
	viewInfo.subresourceRange = VkImageSubresourceRange{ VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 6 * this->arraySize };

	VkImageView view = VK_NULL_HANDLE;
	if (const auto res = vkCreateImageView(this->context->window->device->device, &viewInfo, nullptr, &view); VK_SUCCESS != res)
		throw Utils::Error("Unable to create image view.\n vkCreateImageView() returned %s", Utils::toString(res).c_str());

	this->descriptorView = vk::ImageView(this->context->window->device->device, view);

	// Alter image view info with 2D view type and layerCount 1 for framebuffer views
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	viewInfo.subresourceRange.layerCount = 1;

	this->framebufferViews.clear();

	// For each element in array create 6 image views (6 faces per cubemap)
	for (std::uint32_t element = 0; element < this->arraySize; element++) {
		for (std::uint32_t face = 0; face < 6; face++) {
			std::uint32_t layer = (element * 6) + face;

			viewInfo.subresourceRange.baseArrayLayer = layer;

			VkImageView framebufferView = VK_NULL_HANDLE;
			if (const auto res = vkCreateImageView(this->context->window->device->device, &viewInfo, nullptr, &framebufferView); VK_SUCCESS != res)
				throw Utils::Error("Unable to create image view.\n vkCreateImageView() returned %s", Utils::toString(res).c_str());

			this->framebufferViews.emplace_back(vk::ImageView(this->context->window->device->device, framebufferView));
		}
	}

	TextureBuffer::recreate();
}

vk::ImageView& CubemapArrayDepthTextureBuffer::getImageView() {
	return this->descriptorView;
}

std::vector<vk::ImageView>& CubemapArrayDepthTextureBuffer::getFramebufferViews() {
	return this->framebufferViews;
}
