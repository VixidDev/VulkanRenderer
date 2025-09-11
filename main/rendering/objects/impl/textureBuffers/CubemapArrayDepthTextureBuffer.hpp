#pragma once

#include "../../base/TextureBuffer.hpp"

class CubemapArrayDepthTextureBuffer : public TextureBuffer {
public:
	CubemapArrayDepthTextureBuffer(VulkanContext* context, std::uint32_t arraySize, VkExtent2D* renderExtent = nullptr);

	void recreate();

	// Overridden to return the descriptor view. Downcast and use getFramebufferViews() to get individual face views
	vk::ImageView& getImageView() override;
	// Downcasting will be required to access this method! Will return 6 * number of point lights image views.
	std::vector<vk::ImageView>& getFramebufferViews();
private:
	vk::ImageView descriptorView;
	std::vector<vk::ImageView> framebufferViews;

	std::uint32_t arraySize = 1;
};