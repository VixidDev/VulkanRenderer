#include "WriteToTargetFramebuffer.hpp"

#include "../../../PipelineCreation.hpp"
#include "../../../../vulkan/Swapchain.hpp"

WriteToTargetFramebuffer::WriteToTargetFramebuffer(VulkanWindow* window, TextureBuffer* textureBuffer, RenderPass* renderPass) 
	: textureBuffer(textureBuffer), renderPass(renderPass), Framebuffer(window) 
{
	this->renderExtent = &this->window->getSwapchain()->getExtent();
	this->recreate();
}

void WriteToTargetFramebuffer::recreate() {
	this->framebuffers.clear();

	std::vector<VkImageView> views;
	views.emplace_back(textureBuffer->getImageView().handle);

	createFramebuffers(*this->window, this->framebuffers, this->renderPass->getRenderPassHandle(), views, *this->renderExtent);
}

TextureBuffer* WriteToTargetFramebuffer::getRenderTarget() {
	return this->textureBuffer;
}
