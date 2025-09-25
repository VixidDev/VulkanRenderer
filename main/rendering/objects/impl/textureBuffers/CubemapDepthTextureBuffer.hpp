#pragma once

#include "../../base/TextureBuffer.hpp"

class CubemapDepthTextureBuffer : public TextureBuffer {
public:
	CubemapDepthTextureBuffer(
		VulkanContext* context, 
		VkFormat format = VK_FORMAT_D32_SFLOAT,
		VkSampleCountFlagBits* sampleCount = nullptr, 
		VkExtent2D* renderExtent = nullptr);

	void recreate();

	vk::ImageView& getImageView() override;
	std::vector<vk::ImageView>& getFramebufferViews();
private:
	vk::ImageView descriptorView;
	std::vector<vk::ImageView> framebufferViews;

	bool issuedWarning = false;
};