#include "ForwardFramebuffer.hpp"

#include "../../../PipelineCreation.hpp"

ForwardFramebuffer::ForwardFramebuffer(
	VulkanWindow* window,
	std::map<std::string, _TextureBuffer>* textureBuffers,
	RenderPass* renderPass,
	VkSampleCountFlagBits* sampleCount) : 
	textureBuffers(textureBuffers),
	renderPass(renderPass),
	sampleCount(sampleCount), 
	Framebuffer(window)
{
	this->renderExtent = &this->window->swapchainExtent;

	this->recreate();
}

void ForwardFramebuffer::recreate() {
	this->framebuffers.clear();

	bool usingMSAA = !(*this->sampleCount & VK_SAMPLE_COUNT_1_BIT);

	std::vector<VkImageView> views;
	views.emplace_back(this->textureBuffers->at("colour")->getImageView().handle);
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