#pragma once

#include "../../base/TextureBuffer.hpp"

class CubemapTextureBuffer : public TextureBuffer {
public:
	CubemapTextureBuffer(
		VulkanContext* context, 
		VkFormat format,
		VkExtent2D* renderExtent = nullptr,
		bool skipIndividualImageViews = false,
		bool exemptFromRecreation = false);

	void recreate();

	vk::ImageView& getImageView() override;
	std::vector<vk::ImageView>& getFramebufferViews();
private:
	bool skipIndividualImageViews = false;
	bool exemptFromRecreation = false;

	vk::ImageView descriptorView;
	std::vector<vk::ImageView> framebufferViews;

	VkImageAspectFlags aspectFlags = VK_IMAGE_ASPECT_COLOR_BIT;
	VkImageUsageFlags usageFlags = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
};