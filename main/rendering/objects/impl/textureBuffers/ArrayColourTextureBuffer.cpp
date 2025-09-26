#include "ArrayColourTextureBuffer.hpp"

#include "../../../PipelineCreation.hpp"
#include "../../../../vulkan/VulkanContext.hpp"
#include "../../../../vulkan/VulkanDevice.hpp"

#include "Error.hpp"
#include "toString.hpp"

ArrayColourTextureBuffer::ArrayColourTextureBuffer(
	VulkanContext* context,
	std::uint32_t arraySize,
	VkFormat format,
	VkExtent2D* renderExtent
) : ArrayTextureBuffer(context) 
{
	this->arraySize = arraySize;
	this->format = format;

	if (!renderExtent)
		this->renderExtent = &this->context->window->swapchainExtent;
	else
		this->renderExtent = renderExtent;

	this->recreate();
}

void ArrayColourTextureBuffer::recreate() {
	// Create VkImage
	VkImageCreateInfo imageInfo = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = this->format,
		.extent = { this->renderExtent->width, this->renderExtent->height, 1 },
		.mipLevels = 1,
		.arrayLayers = this->arraySize,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
	};

	// Check if array size is 0 and instead create a dummy texture
	if (this->arraySize == 0) {
		imageInfo.arrayLayers = 1;
		imageInfo.extent = { 1, 1, 1 };
	}

	VmaAllocationCreateInfo allocInfo{};
	allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

	VkImage image = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;

	if (const auto res = vmaCreateImage(this->context->allocator->allocator, &imageInfo, &allocInfo, &image, &allocation, nullptr); VK_SUCCESS != res) {
		std::fprintf(stderr, "Unable to allocate depth buffer image.\n vmaCreateImage() returned %s\n", Utils::toString(res).c_str());
		throw Utils::Error("Unable to allocate depth buffer image.\n vmaCreateImage() returned %s\n", Utils::toString(res).c_str());
	}

	this->image = vk::Image(this->context->allocator->allocator, image, allocation);

	VkImageViewCreateInfo viewInfo = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = this->image.image,
		.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY,
		.format = this->format,
		.components = VkComponentMapping{},
		.subresourceRange = VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, this->arraySize }
	};

	if (this->arraySize == 0) {
		viewInfo.subresourceRange.layerCount = 1;
	}

	VkImageView view = VK_NULL_HANDLE;
	if (const auto res = vkCreateImageView(this->context->window->device->device, &viewInfo, nullptr, &view); VK_SUCCESS != res)
		throw Utils::Error("Unable to create image view.\n vkCreateImageView() returned %s", Utils::toString(res).c_str());

	this->descriptorView = vk::ImageView(this->context->window->device->device, view);

	// If array size is 0, we most likely are not writing to this texture via a framebuffer
	if (this->arraySize == 0) {
		TextureBuffer::recreate();
		return;
	}

	// Alter image view info with 2D view type and layerCount 1 for framebuffer views
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	viewInfo.subresourceRange.layerCount = 1;

	this->framebufferViews.clear();

	// For each element in array create an image view
	for (std::uint32_t element = 0; element < this->arraySize; element++) {
		viewInfo.subresourceRange.baseArrayLayer = element;

		VkImageView framebufferView = VK_NULL_HANDLE;
		if (const auto res = vkCreateImageView(this->context->window->device->device, &viewInfo, nullptr, &framebufferView); VK_SUCCESS != res)
			throw Utils::Error("Unable to create image view.\n vkCreateImageView() returned %s", Utils::toString(res).c_str());

		this->framebufferViews.emplace_back(vk::ImageView(this->context->window->device->device, framebufferView));
	}

	TextureBuffer::recreate();
}

vk::ImageView& ArrayColourTextureBuffer::getImageView() {
	return this->descriptorView;
}