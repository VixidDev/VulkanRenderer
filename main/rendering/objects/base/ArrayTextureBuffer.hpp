#pragma once

#include "TextureBuffer.hpp"

class ArrayTextureBuffer : public TextureBuffer {
public:
	ArrayTextureBuffer() = default;
	ArrayTextureBuffer(
		VulkanContext* context, 
		bool isCubemap,
		std::uint32_t arraySize, 
		VkFormat format = VK_FORMAT_R16G16B16A16_SFLOAT, 
		VkExtent2D* renderExtent = nullptr);

	void recreate() override;
	
	std::vector<vk::ImageView>& getFramebufferViews();
	std::uint32_t getArraySize();
protected:
	std::vector<vk::ImageView> framebufferViews;
	std::uint32_t arraySize = 1;

	bool isCubemap = false;

	VkImageAspectFlags aspectFlags = VK_IMAGE_ASPECT_COLOR_BIT;
	VkImageUsageFlags usageFlags = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
};