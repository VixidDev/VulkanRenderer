#include "DeferredShadingFramebuffer.hpp"

#include "../../../PipelineCreation.hpp"
#include "../../../../vulkan/Swapchain.hpp"

DeferredShadingFramebuffer::DeferredShadingFramebuffer(
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

void DeferredShadingFramebuffer::recreate() {
	this->framebuffers.clear();

	std::vector<VkImageView> views;
	views.emplace_back(this->textureBuffers->at("HDR")->getImageView().handle);
	views.emplace_back(this->textureBuffers->at("brightness")->getImageView().handle);

	createFramebuffers(*this->window, this->framebuffers, this->renderPass->getRenderPassHandle(), views, *this->renderExtent);
}