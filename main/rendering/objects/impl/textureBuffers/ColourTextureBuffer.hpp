#pragma once

#include "../../base/TextureBuffer.hpp"

class ColourTextureBuffer : public TextureBuffer {
public:
	ColourTextureBuffer(VulkanContext* context, VkSampleCountFlagBits* sampleCount, VkExtent2D* renderExtent = nullptr);

	void recreate();
private:
};