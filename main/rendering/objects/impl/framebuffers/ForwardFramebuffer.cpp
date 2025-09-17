#include "ForwardFramebuffer.hpp"

#include "../../../PipelineCreation.hpp"

ForwardFramebuffer::ForwardFramebuffer(
	VulkanWindow* window,
	std::map<std::string, _TextureBuffer>* textureBuffers,
	RenderPass* renderPass,
	VkSampleCountFlagBits* sampleCount) : Framebuffer(window)
{
	this->textureBuffers = textureBuffers;
	this->renderPass = renderPass;
	this->sampleCount = sampleCount;

	this->renderExtent = &this->window->swapchainExtent;

	this->recreate();
}

void ForwardFramebuffer::recreate() {
	this->framebuffers.clear();

	bool usingMSAA = !(*this->sampleCount & VK_SAMPLE_COUNT_1_BIT);

	std::vector<VkImageView> forwardViews;
	forwardViews.emplace_back(this->textureBuffers->at("depth")->getImageView().handle);

	std::vector<VkImageView> forwardMSAAViews;
	//forwardMSAAViews.emplace_back(this->textureBuffers->at("multisampleColour")->getImageView().handle);
	//forwardMSAAViews.emplace_back(this->textureBuffers->at("multisampleDepth")->getImageView().handle);

	createFramebuffers(
		*this->window,
		this->framebuffers,
		this->renderPass->getRenderPassHandle(),
		usingMSAA ? forwardMSAAViews : forwardViews,
		*this->renderExtent);
}