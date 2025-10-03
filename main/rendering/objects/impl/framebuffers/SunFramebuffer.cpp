#include "SunFramebuffer.hpp"

#include "../../../PipelineCreation.hpp"
#include "../../../../vulkan/Swapchain.hpp"

SunFramebuffer::SunFramebuffer(
	VulkanWindow* window,
	std::map<std::string, _TextureBuffer>* textureBuffers,
	RenderPass* renderPass,
	VkSampleCountFlagBits* sampleCount) : Framebuffer(window) 
{
	this->textureBuffers = textureBuffers;
	this->renderPass = renderPass;
	this->sampleCount = sampleCount;

	this->renderExtent = &this->window->getSwapchain()->getExtent();

	this->recreate();
}

void SunFramebuffer::recreate() {
	this->framebuffers.clear();

	std::vector<VkImageView> views;
	views.emplace_back(this->textureBuffers->at("sunView")->getImageView().handle);
	views.emplace_back(this->textureBuffers->at("depth")->getImageView().handle);

	createFramebuffers(*this->window, this->framebuffers, this->renderPass->getRenderPassHandle(), views, *this->renderExtent);
}