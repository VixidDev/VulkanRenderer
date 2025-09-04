#include "Framebuffer.hpp"

Framebuffer::Framebuffer(VulkanWindow* window) : window(window) {}

void Framebuffer::recreate() {}

VkFramebuffer Framebuffer::getHandle(std::uint32_t imageIndex) {
	return this->framebuffers.at(imageIndex).handle;
}

VkExtent2D Framebuffer::getRenderExtent() {
	return *this->renderExtent;
}