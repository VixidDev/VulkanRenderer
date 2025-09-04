#pragma once

#include <volk/volk.h>
#include <vk_mem_alloc.h>

#include <utility>

#include <cassert>

#include "../VulkanContext.hpp"
#include "VkObjects.hpp"

namespace vk {

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

	Image loadImageTexture(const char* path, const VulkanContext& context, VkCommandPool cmdPool, const VulkanAllocator& allocator, VkFormat format, std::uint8_t channels);
	Image createImageTexture(const VulkanAllocator& allocator, std::uint32_t width, std::uint32_t height, VkFormat format, VkImageUsageFlags usageFlags = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
	Image createDummyImage(const VulkanContext& context, const VulkanAllocator& allocator, VkCommandPool cmdPool, VkFormat format);

	ImageView createImageViewTexture(const VulkanContext& context, VkImage image, VkFormat format);

	std::uint32_t computeMipLevelCount(std::uint32_t width, std::uint32_t height);

}
