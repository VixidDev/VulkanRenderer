#pragma once

#include "../../base/ArrayTextureBuffer.hpp"

class ArrayColourTextureBuffer : public ArrayTextureBuffer {
public:
	ArrayColourTextureBuffer(
		VulkanContext* context, 
		std::uint32_t arraySize, 
		VkFormat format = VK_FORMAT_R16G16B16A16_SFLOAT, 
		VkExtent2D* renderExtent = nullptr);

	void recreate();

	// Overridden to return the descriptor view. Downcast to ArrayTextureBuffer and use getFramebufferViews() to get individual face views
	vk::ImageView& getImageView() override;
private:
	vk::ImageView descriptorView;
};