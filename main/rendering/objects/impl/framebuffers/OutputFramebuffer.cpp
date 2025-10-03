#include "OutputFramebuffer.hpp"

#include "../../../PipelineCreation.hpp"
#include "../../../../vulkan/Swapchain.hpp"

OutputFramebuffer::OutputFramebuffer(
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

void OutputFramebuffer::recreate() {
	this->framebuffers.clear();

	std::vector<VkImageView> views;
	views.emplace_back(this->textureBuffers->at("colour")->getImageView().handle);

	createFramebuffers(*this->window, this->framebuffers, this->renderPass->getRenderPassHandle(), views, *this->renderExtent);
}