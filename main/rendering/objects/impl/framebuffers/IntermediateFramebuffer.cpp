#include "IntermediateFramebuffer.hpp"

#include "../../../PipelineCreation.hpp"
#include "../../../../vulkan/Swapchain.hpp"

IntermediateFramebuffer::IntermediateFramebuffer(
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

void IntermediateFramebuffer::recreate() {
	this->framebuffers.clear();

	std::vector<VkImageView> views;
	views.emplace_back(this->textureBuffers->at("intermediate")->getImageView().handle);

	createFramebuffers(*this->window, this->framebuffers, this->renderPass->getRenderPassHandle(), views, *this->renderExtent);
}