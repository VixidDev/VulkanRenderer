#include "NoiseTextureBuffer.hpp"

#include <random>

#include "Error.hpp"
#include "toString.hpp"
#include "../../../PipelineCreation.hpp"
#include "../../../../vulkan/VulkanContext.hpp"
#include "../../../../vulkan/VulkanDevice.hpp"
#include "../../../../vulkan/Swapchain.hpp"
#include "../../../../vulkan/objects/VkBuffer.hpp"
#include "../../../../vulkan/VkUtils.hpp"

#include <glm/vec2.hpp>

namespace {
	std::uniform_real_distribution<float> randomFloats(0.0f, 1.0f);
	std::default_random_engine randomEngine;
}

NoiseTextureBuffer::NoiseTextureBuffer(VulkanContext* context) : TextureBuffer(context) {
	this->format = VK_FORMAT_R16G16_SNORM;

	// Create random noise
	std::vector<glm::vec2> noise(4 * 4);
	for (glm::vec2& sample : noise) {
		sample = glm::vec2(
			randomFloats(randomEngine) * 2.0f - 1.0f, 
			randomFloats(randomEngine) * 2.0f - 1.0f);
	}

	const VulkanAllocator& allocator = *this->context->allocator;

	std::size_t bufferSize = noise.size() * sizeof(glm::vec2);

	// Create staging buffer
	vk::Buffer staging = vk::createBuffer(
		allocator,
		bufferSize,
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

	// Map staging buffer memory
	void* ptr = nullptr;
	if (const VkResult res = vmaMapMemory(allocator.allocator, staging.allocation, &ptr); VK_SUCCESS != res)
		throw Utils::Error("Unable to map memory\nvmaMapMemory() returned %s\n", Utils::toString(res).c_str());
	std::memcpy(ptr, noise.data(), bufferSize);
	vmaUnmapMemory(allocator.allocator, staging.allocation);

	// Create vk::Image
	vk::Image image = vk::createImage(allocator, 4, 4, this->format, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, false);

	VkCommandPool cmdPool = this->context->window->getDevice()->getCmdPool();
	VkCommandBuffer cbuff = VkUtils::createCommandBuffer(*this->context->window, cmdPool);
	VkUtils::beginCommandBuffer(cbuff);

	// Transition to TRANSFER_DST_OPTIMAL
	VkUtils::imageBarrier(cbuff, image.image,
		/* srcAccessMask */ 0, /* dstAccessMask */ VK_ACCESS_TRANSFER_WRITE_BIT,
		/* srcLayout     */ VK_IMAGE_LAYOUT_UNDEFINED, /* dstLayout */ VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		/* srcStageMask  */ VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, /* dstStageMask */ VK_PIPELINE_STAGE_TRANSFER_BIT,
		VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 });

	// Copy buffer into image
	VkBufferImageCopy copy;
	copy.bufferOffset = 0;
	copy.bufferRowLength = 0;
	copy.bufferImageHeight = 0;
	copy.imageSubresource = VkImageSubresourceLayers{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
	copy.imageOffset = VkOffset3D{ 0, 0, 0 };
	copy.imageExtent = VkExtent3D{ 4, 4, 1 };

	vkCmdCopyBufferToImage(cbuff, staging.buffer, image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

	// Transition to SHADER_READ_ONLY_OPTIMAL
	VkUtils::imageBarrier(cbuff, image.image,
		/* srcAccessMask */ VK_ACCESS_TRANSFER_WRITE_BIT, /* dstAccessMask */ VK_ACCESS_SHADER_READ_BIT,
		/* srcLayout     */ VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, /* dstLayout */ VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		/* srcStageMask  */ VK_PIPELINE_STAGE_TRANSFER_BIT, /* dstStageMask */ VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
		VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 });

	VkUtils::endAndSubmitCommandBuffer(*this->context->window, cbuff);

	// Move into object variable
	this->image = std::move(image);

	// Create image view for noise texture
	VkImageViewCreateInfo viewInfo{};
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.image = this->image.image;
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	viewInfo.format = format;
	viewInfo.components = VkComponentMapping{};
	viewInfo.subresourceRange = VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, 1 };

	VkImageView view = VK_NULL_HANDLE;
	if (const auto res = vkCreateImageView(this->context->window->getDevice()->getDevice(), &viewInfo, nullptr, &view); VK_SUCCESS != res)
		throw Utils::Error("Unable to create image view\n vkCreateImageView() returned %s", Utils::toString(res).c_str());

	this->imageView = vk::ImageView(this->context->window->getDevice()->getDevice(), view);
}

void NoiseTextureBuffer::recreate() {
	// This texture does not need to ever be recreated once initialised
}