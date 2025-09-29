#include "BlurFramebuffer.hpp"

#include "../../../PipelineCreation.hpp"

BlurFramebuffer::BlurFramebuffer(
	VulkanWindow* window,
	std::map<std::string, _TextureBuffer>* textureBuffers,
	RenderPass* renderPass
) : textureBuffers(textureBuffers),
	renderPass(renderPass),
	Framebuffer(window) 
{
	this->renderExtent = &this->window->swapchainExtent;

	this->recreate();
}

void BlurFramebuffer::recreate() {
	this->framebuffers.clear();

	std::vector<VkImageView> views;
	views.emplace_back(this->textureBuffers->at("blurOutput")->getImageView().handle);

	createFramebuffers(*this->window, this->framebuffers, this->renderPass->getRenderPassHandle(), views, *this->renderExtent);
}