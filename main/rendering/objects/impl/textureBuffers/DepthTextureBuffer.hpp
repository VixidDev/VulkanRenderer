#pragma once

#include "../../base/TextureBuffer.hpp"

class DepthTextureBuffer : public TextureBuffer {
public:
	DepthTextureBuffer(
		VulkanContext* context, 
		VkFormat format = VK_FORMAT_D32_SFLOAT,
		VkExtent2D* renderExtent = nullptr);

	void recreate();
private:
};