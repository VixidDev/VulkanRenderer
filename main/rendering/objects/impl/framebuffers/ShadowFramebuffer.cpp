#include "ShadowFramebuffer.hpp"

#include "../../../PipelineCreation.hpp"

ShadowFramebuffer::ShadowFramebuffer(
	VulkanWindow* window,
	std::map<std::string, _TextureBuffer>* textureBuffers,
	RenderPass* renderPass,
	VkExtent2D* shadowMapResolution) : Framebuffer(window) {
	this->textureBuffers = textureBuffers;
	this->renderPass = renderPass;

	this->renderExtent = shadowMapResolution;

	this->recreate();
}

void ShadowFramebuffer::recreate() {
	this->framebuffers.clear();

	std::vector<VkImageView> shadowViews;
	shadowViews.emplace_back(this->textureBuffers->at("shadowDepth")->getImageView().handle);
	shadowViews.emplace_back(this->textureBuffers->at("debugLinearDepth")->getImageView().handle);

	createFramebuffers(*this->window, this->framebuffers, this->renderPass->getRenderPassHandle(), shadowViews, *this->renderExtent, true);
}