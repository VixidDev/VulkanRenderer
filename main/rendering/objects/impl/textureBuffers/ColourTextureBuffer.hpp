#pragma once

#include "../../base/TextureBuffer.hpp"

class ColourTextureBuffer : public TextureBuffer {
public:
	ColourTextureBuffer(
		VulkanContext* context, 
		VkFormat format = VK_FORMAT_R16G16B16A16_SFLOAT,
		VkSampleCountFlagBits* sampleCount = nullptr, 
		VkExtent2D* renderExtent = nullptr);

	void recreate();
private:
};