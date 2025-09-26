#pragma once

#include "../../base/TextureBuffer.hpp"

class ShadowDepthTextureBuffer : public TextureBuffer {
public:
	ShadowDepthTextureBuffer(VulkanContext* context, VkSampleCountFlagBits* sampleCount, VkExtent2D* renderExtent = nullptr);

	void recreate();
private:
};