#include "ArrayTextureBuffer.hpp"

ArrayTextureBuffer::ArrayTextureBuffer(VulkanContext* context) : TextureBuffer(context) {}

std::vector<vk::ImageView>& ArrayTextureBuffer::getFramebufferViews() {
	return this->framebufferViews;
}

std::uint32_t ArrayTextureBuffer::getArraySize() {
	return this->arraySize;
}