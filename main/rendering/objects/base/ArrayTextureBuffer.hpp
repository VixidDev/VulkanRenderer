#pragma once

#include "TextureBuffer.hpp"

class ArrayTextureBuffer : public TextureBuffer {
public:
	ArrayTextureBuffer() = default;
	ArrayTextureBuffer(VulkanContext* context);

	std::vector<vk::ImageView>& getFramebufferViews();
	std::uint32_t getArraySize();
protected:
	std::vector<vk::ImageView> framebufferViews;
	std::uint32_t arraySize = 1;
};