#include "DeferredFramebuffer.hpp"

#include "../../../PipelineCreation.hpp"

DeferredFramebuffer::DeferredFramebuffer(
	VulkanWindow* window,
	std::map<std::string, _TextureBuffer>* textureBuffers,
	RenderPass* renderPass,
	VkSampleCountFlagBits* sampleCount
) : textureBuffers(textureBuffers),
	renderPass(renderPass),
	sampleCount(sampleCount),
	Framebuffer(window) 
{
	this->renderExtent = &this->window->swapchainExtent;

	this->recreate();
}

void DeferredFramebuffer::recreate() {
	this->framebuffers.clear();

	bool usingMSAA = !(*this->sampleCount & VK_SAMPLE_COUNT_1_BIT);

	std::vector<VkImageView> views;
	views.emplace_back(this->textureBuffers->at("colour")->getImageView().handle);
	views.emplace_back(this->textureBuffers->at("brightness")->getImageView().handle);
	views.emplace_back(this->textureBuffers->at("gBuffer1")->getImageView().handle);
	views.emplace_back(this->textureBuffers->at("gBuffer2")->getImageView().handle);
	views.emplace_back(this->textureBuffers->at("gBuffer3")->getImageView().handle);
	views.emplace_back(this->textureBuffers->at("depth")->getImageView().handle);

	std::vector<VkImageView> MSAAViews;
	//MSAAViews.emplace_back(this->textureBuffers->at("multisampleColour")->getImageView().handle);
	//MSAAViews.emplace_back(this->textureBuffers->at("multisampleDepth")->getImageView().handle);

	createFramebuffers(
		*this->window,
		this->framebuffers,
		this->renderPass->getRenderPassHandle(),
		usingMSAA ? MSAAViews : views,
		*this->renderExtent);
}