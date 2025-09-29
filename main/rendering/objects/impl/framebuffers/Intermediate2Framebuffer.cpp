#include "Intermediate2Framebuffer.hpp"

#include "../../../PipelineCreation.hpp"

Intermediate2Framebuffer::Intermediate2Framebuffer(
	VulkanWindow* window,
	std::map<std::string, _TextureBuffer>* textureBuffers,
	RenderPass* renderPass
) : textureBuffers(textureBuffers),
	renderPass(renderPass),
	Framebuffer(window) {
	this->renderExtent = &this->window->swapchainExtent;

	this->recreate();
}

void Intermediate2Framebuffer::recreate() {
	this->framebuffers.clear();

	std::vector<VkImageView> views;
	views.emplace_back(this->textureBuffers->at("intermediate2")->getImageView().handle);

	createFramebuffers(*this->window, this->framebuffers, this->renderPass->getRenderPassHandle(), views, *this->renderExtent);
}