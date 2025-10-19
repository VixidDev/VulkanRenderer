#pragma once

#include <utility>
#include <cassert>

#include <volk/volk.h>
#include <vk_mem_alloc.h>
#include <stb_image.h>

#include "../VulkanContext.hpp"
#include "VkObjects.hpp"

namespace vk {

	struct ImageData {
		stbi_uc* data = nullptr;
		int width;
		int height;
		VkFormat format;
	};

	class Image {
	public:
		Image() noexcept, ~Image();

		explicit Image(VmaAllocator allocator, VkImage image = VK_NULL_HANDLE, VmaAllocation allocation = VK_NULL_HANDLE) noexcept;

		Image(Image const&) = delete;
		Image& operator= (Image const&) = delete;

		Image(Image&&) noexcept;
		Image& operator = (Image&&) noexcept;

	public:
		VkImage image = VK_NULL_HANDLE;
		VmaAllocation allocation = VK_NULL_HANDLE;

	private:
		VmaAllocator mAllocator = VK_NULL_HANDLE;
	};

	Image loadImage(ImageData& imageData, const VulkanContext& context);
	Image createImage(const VulkanAllocator& allocator, std::uint32_t width, std::uint32_t height, VkFormat format, VkImageUsageFlags usageFlags = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, bool useMips = true);
	Image createDummyImage(const VulkanContext& context, VkFormat format);

	ImageView createImageView(const VulkanContext& context, VkImage image, VkFormat format);

	std::uint32_t computeMipLevelCount(std::uint32_t width, std::uint32_t height);

}
