#include "VkImage.hpp"

#include <bit>
#include <limits>
#include <vector>
#include <utility>
#include <algorithm>

#include <cstdio>
#include <cassert>
#include <cstring> // for std::memcpy()

#include <stb_image.h>

#include "Error.hpp"
#include "VkBuffer.hpp"
#include "toString.hpp"
#include "../VkUtils.hpp"
#include "../VulkanDevice.hpp"

#include <iostream>

namespace vk {

	Image::Image() noexcept = default;

	Image::~Image() {
		if (VK_NULL_HANDLE != image) {
			assert(VK_NULL_HANDLE != mAllocator);
			assert(VK_NULL_HANDLE != allocation);
			vmaDestroyImage(mAllocator, image, allocation);
		}
	}

	Image::Image(VmaAllocator aAllocator, VkImage aImage, VmaAllocation aAllocation) noexcept
		: image(aImage)
		, allocation(aAllocation)
		, mAllocator(aAllocator) {}

	Image::Image(Image&& aOther) noexcept
		: image(std::exchange(aOther.image, VK_NULL_HANDLE))
		, allocation(std::exchange(aOther.allocation, VK_NULL_HANDLE))
		, mAllocator(std::exchange(aOther.mAllocator, VK_NULL_HANDLE)) {}
	Image& Image::operator=(Image&& aOther) noexcept {
		std::swap(image, aOther.image);
		std::swap(allocation, aOther.allocation);
		std::swap(mAllocator, aOther.mAllocator);
		return *this;
	}

	Image loadImageTexture(char const* path, const VulkanContext& context, VkFormat format, std::uint8_t channels) {
		const VulkanAllocator& allocator = *context.allocator;
		VkCommandPool cmdPool = context.window->device->cmdPool;

		stbi_set_flip_vertically_on_load(1);

		int baseWidthi, baseHeighti, baseChannelsi;
		stbi_uc* data = stbi_load(path, &baseWidthi, &baseHeighti, &baseChannelsi, 4);
		// std::cout << "Desired: " << channels << " baseChannelsi: " << baseChannelsi << std::endl;

		if (!data)
			throw Utils::Error("%s: Unable to load texture base image (%s)", path, 0, stbi_failure_reason());

		const auto baseWidth = std::uint32_t(baseWidthi);
		const auto baseHeight = std::uint32_t(baseHeighti);

		const auto sizeInBytes = baseWidth * baseHeight * 4;

		auto staging = createBuffer(
			allocator,
			sizeInBytes,
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
		);

		void* sptr = nullptr;
		if (const auto res = vmaMapMemory(allocator.allocator, staging.allocation, &sptr); VK_SUCCESS != res)
			throw Utils::Error("Unable to map memory\n vmaMapMemory() returned %s", Utils::toString(res).c_str());

		std::memcpy(sptr, data, sizeInBytes);
		vmaUnmapMemory(allocator.allocator, staging.allocation);

		stbi_image_free(data);

		Image ret = createImageTexture(
			allocator,
			baseWidth,
			baseHeight,
			format,
			VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
		);
		
		VkCommandBuffer cbuff = VkUtils::createCommandBuffer(*context.window, cmdPool);

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = 0;
		beginInfo.pInheritanceInfo = nullptr;

		if (const auto res = vkBeginCommandBuffer(cbuff, &beginInfo); VK_SUCCESS != res)
			throw Utils::Error("Unable to begin command buffer\n vkBeginCommandBuffer() returned %s", Utils::toString(res).c_str());

		const auto mipLevels = computeMipLevelCount(baseWidth, baseHeight);

		VkUtils::imageBarrier(
			cbuff,
			ret.image,
			0,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_IMAGE_LAYOUT_UNDEFINED,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VkImageSubresourceRange{
				VK_IMAGE_ASPECT_COLOR_BIT,
				0,
				mipLevels,
				0,
				1
			}
		);

		VkBufferImageCopy copy;
		copy.bufferOffset = 0;
		copy.bufferRowLength = 0;
		copy.bufferImageHeight = 0;
		copy.imageSubresource = VkImageSubresourceLayers{
			VK_IMAGE_ASPECT_COLOR_BIT,
			0,
			0,
			1
		};
		copy.imageOffset = VkOffset3D{ 0, 0, 0 };
		copy.imageExtent = VkExtent3D{ baseWidth, baseHeight, 1 };

		vkCmdCopyBufferToImage(cbuff, staging.buffer, ret.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

		VkUtils::imageBarrier(
			cbuff,
			ret.image,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_TRANSFER_READ_BIT,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VkImageSubresourceRange{
				VK_IMAGE_ASPECT_COLOR_BIT,
				0,
				1,
				0,
				1
			}
		);

		uint32_t width = baseWidth, height = baseHeight;

		for (std::uint32_t level = 1; level < mipLevels; ++level) {
			VkImageBlit blit{};
			blit.srcSubresource = VkImageSubresourceLayers{
				VK_IMAGE_ASPECT_COLOR_BIT,
				level - 1,
				0,
				1
			};
			blit.srcOffsets[0] = { 0, 0, 0 };
			blit.srcOffsets[1] = { std::int32_t(width), std::int32_t(height), 1 };

			width >>= 1; if (width == 0) width = 1;
			height >>= 1; if (height == 0) height = 1;

			blit.dstSubresource = VkImageSubresourceLayers{
				VK_IMAGE_ASPECT_COLOR_BIT,
				level,
				0,
				1
			};
			blit.dstOffsets[0] = { 0, 0, 0 };
			blit.dstOffsets[1] = { std::int32_t(width), std::int32_t(height), 1 };

			vkCmdBlitImage(
				cbuff,
				ret.image,
				VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				ret.image,
				VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				1,
				&blit,
				VK_FILTER_LINEAR
			);

			VkUtils::imageBarrier(
				cbuff,
				ret.image,
				VK_ACCESS_TRANSFER_WRITE_BIT,
				VK_ACCESS_TRANSFER_READ_BIT,
				VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VkImageSubresourceRange{
					VK_IMAGE_ASPECT_COLOR_BIT,
					level,
					1,
					0,
					1
				}
			);
		}

		VkUtils::imageBarrier(
			cbuff,
			ret.image,
			VK_ACCESS_TRANSFER_READ_BIT,
			VK_ACCESS_SHADER_READ_BIT,
			VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			VkImageSubresourceRange{
				VK_IMAGE_ASPECT_COLOR_BIT,
				0,
				mipLevels,
				0,
				1
			}
		);

		if (const auto res = vkEndCommandBuffer(cbuff); VK_SUCCESS != res)
			throw Utils::Error("Unable to end command buffer\n vkEndCommandBuffer() returned %s", Utils::toString(res).c_str());

		Fence uploadComplete = VkUtils::createFence(*context.window);

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &cbuff;

		if (const auto res = vkQueueSubmit(context.window->graphicsQueue, 1, &submitInfo, uploadComplete.handle); VK_SUCCESS != res)
			throw Utils::Error("Unable to queue submit\n vkQueueSubmit() returned %s", Utils::toString(res).c_str());

		if (const auto res = vkWaitForFences(context.window->device->device, 1, &uploadComplete.handle, VK_TRUE, std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res)
			throw Utils::Error("Unable to wait for fences\n vkWaitForFences() returned %s", Utils::toString(res).c_str());

		vkFreeCommandBuffers(context.window->device->device, cmdPool, 1, &cbuff);

		return ret;
	}

	Image createImageTexture(const VulkanAllocator& allocator, std::uint32_t width, std::uint32_t height, VkFormat format, VkImageUsageFlags usageFlags) {
		const auto mipLevels = computeMipLevelCount(width, height);

		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.format = format;
		imageInfo.extent.width = width;
		imageInfo.extent.height = height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = mipLevels;
		imageInfo.arrayLayers = 1;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.usage = usageFlags;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

		VmaAllocationCreateInfo allocInfo{};
		allocInfo.flags = 0;
		allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

		VkImage image = VK_NULL_HANDLE;
		VmaAllocation allocation = VK_NULL_HANDLE;

		if (const auto res = vmaCreateImage(allocator.allocator, &imageInfo, &allocInfo, &image, &allocation, nullptr); VK_SUCCESS != res)
			throw Utils::Error("Unable to allocate image\n vmaCreateImage() returned %s", Utils::toString(res).c_str());

		return Image(allocator.allocator, image, allocation);
	}

	Image createDummyImage(const VulkanContext& context, VkFormat format) {
		const VulkanAllocator& allocator = *context.allocator;
		VkCommandPool cmdPool = context.window->device->cmdPool;

		std::uint8_t data[4] = { std::uint8_t(255), std::uint8_t(255), std::uint8_t(255), std::uint8_t(255) };

		auto staging = createBuffer(
			allocator,
			4, // 1 byte
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
		);

		void* sptr = nullptr;
		if (const auto res = vmaMapMemory(allocator.allocator, staging.allocation, &sptr); VK_SUCCESS != res)
			throw Utils::Error("Unable to map memory\n vmaMapMemory() returned %s", Utils::toString(res).c_str());

		std::memcpy(sptr, data, 4);
		vmaUnmapMemory(allocator.allocator, staging.allocation);

		Image ret = createImageTexture(allocator, 1, 1, format,
			VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
		);

		VkCommandBuffer cbuff = VkUtils::createCommandBuffer(*context.window, cmdPool);

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = 0;
		beginInfo.pInheritanceInfo = nullptr;

		if (const auto res = vkBeginCommandBuffer(cbuff, &beginInfo); VK_SUCCESS != res)
			throw Utils::Error("Unable to begin command buffer\n vkBeginCommandBuffer() returned %s", Utils::toString(res).c_str());

		VkUtils::imageBarrier(
			cbuff,
			ret.image,
			0,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_IMAGE_LAYOUT_UNDEFINED,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VkImageSubresourceRange{
				VK_IMAGE_ASPECT_COLOR_BIT,
				0,
				1,
				0,
				1
			}
		);

		VkBufferImageCopy copy;
		copy.bufferOffset = 0;
		copy.bufferRowLength = 0;
		copy.bufferImageHeight = 0;
		copy.imageSubresource = VkImageSubresourceLayers{
			VK_IMAGE_ASPECT_COLOR_BIT,
			0,
			0,
			1
		};
		copy.imageOffset = VkOffset3D{ 0, 0, 0 };
		copy.imageExtent = VkExtent3D{ 1, 1, 1 };

		vkCmdCopyBufferToImage(cbuff, staging.buffer, ret.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

		VkUtils::imageBarrier(
			cbuff,
			ret.image,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_TRANSFER_READ_BIT,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VkImageSubresourceRange{
				VK_IMAGE_ASPECT_COLOR_BIT,
				0,
				1,
				0,
				1
			}
		);

		VkUtils::imageBarrier(
			cbuff,
			ret.image,
			VK_ACCESS_TRANSFER_READ_BIT,
			VK_ACCESS_SHADER_READ_BIT,
			VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			VkImageSubresourceRange{
				VK_IMAGE_ASPECT_COLOR_BIT,
				0,
				1,
				0,
				1
			}
		);

		if (const auto res = vkEndCommandBuffer(cbuff); VK_SUCCESS != res)
			throw Utils::Error("Unable to end command buffer\n vkEndCommandBuffer() returned %s", Utils::toString(res).c_str());

		Fence uploadComplete = VkUtils::createFence(*context.window);

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &cbuff;

		if (const auto res = vkQueueSubmit(context.window->graphicsQueue, 1, &submitInfo, uploadComplete.handle); VK_SUCCESS != res)
			throw Utils::Error("Unable to queue submit\n vkQueueSubmit() returned %s", Utils::toString(res).c_str());

		if (const auto res = vkWaitForFences(context.window->device->device, 1, &uploadComplete.handle, VK_TRUE, std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res)
			throw Utils::Error("Unable to wait for fences\n vkWaitForFences() returned %s", Utils::toString(res).c_str());

		vkFreeCommandBuffers(context.window->device->device, cmdPool, 1, &cbuff);

		return ret;
	}

	ImageView createImageViewTexture(const VulkanContext& context, VkImage image, VkFormat format) {
		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = format;
		viewInfo.components = VkComponentMapping{};
		viewInfo.subresourceRange = VkImageSubresourceRange{
			VK_IMAGE_ASPECT_COLOR_BIT,
			0,
			VK_REMAINING_MIP_LEVELS,
			0,
			1
		};

		VkImageView view = VK_NULL_HANDLE;
		if (const auto res = vkCreateImageView(context.window->device->device, &viewInfo, nullptr, &view); VK_SUCCESS != res)
			throw Utils::Error("Unable to create image view\n vkCreateImageView() returned %s", Utils::toString(res).c_str());

		return ImageView(context.window->device->device, view);
	}

	std::uint32_t computeMipLevelCount(std::uint32_t width, std::uint32_t height) {
		std::uint32_t const bits = width | height;
		std::uint32_t const leadingZeros = std::countl_zero(bits);
		return 32 - leadingZeros;
	}

}