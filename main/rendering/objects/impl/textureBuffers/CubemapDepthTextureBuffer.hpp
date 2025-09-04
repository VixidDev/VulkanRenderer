#pragma once

#include "../../base/TextureBuffer.hpp"

class CubemapDepthTextureBuffer : public TextureBuffer {
public:
	CubemapDepthTextureBuffer(VulkanContext* context, VkSampleCountFlagBits* sampleCount, VkExtent2D* renderExtent = nullptr);

	void recreate();

	vk::ImageView& getImageView() override;
	vk::ImageView& getDescriptorView();
	std::vector<vk::ImageView>& getFramebufferViews();
private:
	vk::ImageView descriptorView;
	std::vector<vk::ImageView> framebufferViews;

	bool issuedWarning = false;
};