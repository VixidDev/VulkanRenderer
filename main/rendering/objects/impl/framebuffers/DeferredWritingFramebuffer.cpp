#include "DeferredWritingFramebuffer.hpp"

#include "../../../PipelineCreation.hpp"
#include "../../../../vulkan/Swapchain.hpp"

DeferredWritingFramebuffer::DeferredWritingFramebuffer(
	VulkanWindow* window,
	std::map<std::string, _TextureBuffer>* textureBuffers,
	RenderPass* renderPass
) : textureBuffers(textureBuffers),
	renderPass(renderPass),
	Framebuffer(window) 
{
	this->renderExtent = &this->window->getSwapchain()->getExtent();

	this->recreate();
}

void DeferredWritingFramebuffer::recreate() {
	this->framebuffers.clear();

	std::vector<VkImageView> views;
	views.emplace_back(this->textureBuffers->at("gBuffer1")->getImageView().handle);
	views.emplace_back(this->textureBuffers->at("gBuffer2")->getImageView().handle);
	views.emplace_back(this->textureBuffers->at("gBuffer3")->getImageView().handle);
	views.emplace_back(this->textureBuffers->at("depth")->getImageView().handle);

	createFramebuffers(*this->window, this->framebuffers, this->renderPass->getRenderPassHandle(), views, *this->renderExtent);
}