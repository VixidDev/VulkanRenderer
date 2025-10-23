#include "ArrayTextureBuffer.hpp"

#include <array>

#include "../../PipelineCreation.hpp"
#include "../../../vulkan/VulkanContext.hpp"
#include "../../../vulkan/VulkanDevice.hpp"
#include "../../../vulkan/Swapchain.hpp"

#include "Error.hpp"
#include "toString.hpp"

ArrayTextureBuffer::ArrayTextureBuffer(
	VulkanContext* context, 
	bool isCubemap,
	std::uint32_t arraySize, 
	VkFormat format, 
	VkExtent2D* renderExtent
) : TextureBuffer(context),
	isCubemap(isCubemap)
{
	this->arraySize = arraySize;
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

	if (std::find(depthFormats.begin(), depthFormats.end(), this->format) != std::end(depthFormats)) {
		this->aspectFlags = VK_IMAGE_ASPECT_DEPTH_BIT;
		this->usageFlags |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
	} else {
		this->usageFlags |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
	}

	std::array<VkFormat, 4> stencilFormats = {
		VK_FORMAT_S8_UINT,
		VK_FORMAT_D16_UNORM_S8_UINT,
		VK_FORMAT_D24_UNORM_S8_UINT,
		VK_FORMAT_D32_SFLOAT_S8_UINT
	};

	if (std::find(stencilFormats.begin(), stencilFormats.end(), this->format) != std::end(stencilFormats)) {
		this->aspectFlags |= VK_IMAGE_ASPECT_STENCIL_BIT;
		this->usageFlags |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
	}

	this->recreate();
}

void ArrayTextureBuffer::recreate() {
	VkImageCreateFlags imageFlags = this->isCubemap ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0;
	std::uint32_t arrayLayers = this->isCubemap ? 6 * this->arraySize : this->arraySize;

	// Create VkImage
	VkImageCreateInfo imageInfo = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.flags = imageFlags,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = this->format,
		.extent = { this->renderExtent->width, this->renderExtent->height, 1 },
		.mipLevels = 1,
		.arrayLayers = arrayLayers,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = this->usageFlags,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
	};

	// Check if array size is 0 and instead create a dummy texture
	if (this->arraySize == 0) {
		imageInfo.arrayLayers = this->isCubemap ? 6 : 1;
		imageInfo.extent = { 1, 1, 1 };
	}

	VmaAllocationCreateInfo allocInfo{};
	allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

	VkImage image = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;

	if (const VkResult res = vmaCreateImage(this->context->allocator->allocator, &imageInfo, &allocInfo, &image, &allocation, nullptr); VK_SUCCESS != res)
		throw Utils::Error("Unable to allocate depth buffer image.\n vmaCreateImage() returned %s\n", Utils::toString(res).c_str());

	this->image = vk::Image(this->context->allocator->allocator, image, allocation);

	VkImageViewType imageViewType = this->isCubemap ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY : VK_IMAGE_VIEW_TYPE_2D_ARRAY;

	VkImageViewCreateInfo viewInfo = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = this->image.image,
		.viewType = imageViewType,
		.format = this->format,
		.components = VkComponentMapping{},
		.subresourceRange = VkImageSubresourceRange{ this->aspectFlags, 0, 1, 0, arrayLayers }
	};

	if (this->arraySize == 0)
		viewInfo.subresourceRange.layerCount = this->isCubemap ? 6 : 1;

	VkImageView view = VK_NULL_HANDLE;
	if (const VkResult res = vkCreateImageView(this->context->window->getDevice()->getDevice(), &viewInfo, nullptr, &view); VK_SUCCESS != res)
		throw Utils::Error("Unable to create image view.\n vkCreateImageView() returned %s", Utils::toString(res).c_str());

	this->imageView = vk::ImageView(this->context->window->getDevice()->getDevice(), view);

	// If array size is 0, we most likely are not writing to this texture via a framebuffer
	if (this->arraySize == 0) {
		TextureBuffer::recreate();
		return;
	}

	// Alter image view info with 2D view type and layerCount 1 for framebuffer views
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	viewInfo.subresourceRange.layerCount = 1;

	this->framebufferViews.clear();

	// More image views need to be made for cubemaps
	if (this->isCubemap) {
		// For each element in array create 6 image views (6 faces per cubemap)
		for (std::uint32_t element = 0; element < this->arraySize; element++) {
			for (std::uint32_t face = 0; face < 6; face++) {
				std::uint32_t layer = (element * 6) + face;

				viewInfo.subresourceRange.baseArrayLayer = layer;

				VkImageView framebufferView = VK_NULL_HANDLE;
				if (const auto res = vkCreateImageView(this->context->window->getDevice()->getDevice(), &viewInfo, nullptr, &framebufferView); VK_SUCCESS != res)
					throw Utils::Error("Unable to create image view.\n vkCreateImageView() returned %s", Utils::toString(res).c_str());

				this->framebufferViews.emplace_back(vk::ImageView(this->context->window->getDevice()->getDevice(), framebufferView));
			}
		}
	} else {
		// For each element in array create an image view
		for (std::uint32_t element = 0; element < this->arraySize; element++) {
			viewInfo.subresourceRange.baseArrayLayer = element;

			VkImageView framebufferView = VK_NULL_HANDLE;
			if (const auto res = vkCreateImageView(this->context->window->getDevice()->getDevice(), &viewInfo, nullptr, &framebufferView); VK_SUCCESS != res)
				throw Utils::Error("Unable to create image view.\n vkCreateImageView() returned %s", Utils::toString(res).c_str());

			this->framebufferViews.emplace_back(vk::ImageView(this->context->window->getDevice()->getDevice(), framebufferView));
		}
	}

	TextureBuffer::recreate();
}

std::vector<vk::ImageView>& ArrayTextureBuffer::getFramebufferViews() {
	return this->framebufferViews;
}

std::uint32_t ArrayTextureBuffer::getArraySize() {
	return this->arraySize;
}