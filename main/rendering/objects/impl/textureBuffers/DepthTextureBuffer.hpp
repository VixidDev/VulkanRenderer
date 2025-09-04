#pragma once

#include "../../base/TextureBuffer.hpp"

class DepthTextureBuffer : public TextureBuffer {
public:
	DepthTextureBuffer(VulkanContext* context, VkSampleCountFlagBits* sampleCount, VkExtent2D* renderExtent = nullptr);

	void recreate();
};