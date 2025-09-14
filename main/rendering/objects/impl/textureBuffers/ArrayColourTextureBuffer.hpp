#pragma once

#include "../../base/ArrayTextureBuffer.hpp"

class ArrayColourTextureBuffer : public ArrayTextureBuffer {
public:
	ArrayColourTextureBuffer(VulkanContext* context, std::uint32_t arraySize, VkExtent2D* renderExtent = nullptr);

	void recreate();

	// Overridden to return the descriptor view. Downcast to ArrayTextureBuffer and use getFramebufferViews() to get individual face views
	vk::ImageView& getImageView() override;
private:
	vk::ImageView descriptorView;
};