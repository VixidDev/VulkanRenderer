#include "ShadowFramebuffer.hpp"

#include "../../../PipelineCreation.hpp"

ShadowFramebuffer::ShadowFramebuffer(
	VulkanWindow* window,
	std::initializer_list<TextureBuffer*> textureBuffers,
	RenderPass* renderPass,
	VkExtent2D* shadowMapResolution
) : textureBuffers(textureBuffers),
	renderPass(renderPass),
	Framebuffer(window) 
{
	this->renderExtent = shadowMapResolution;

	this->recreate();
}

void ShadowFramebuffer::recreate() {
	this->framebuffers.clear();

	std::vector<VkImageView> shadowViews;
	for (std::size_t i = 0; i < this->textureBuffers.size(); i++) {
		shadowViews.emplace_back(this->textureBuffers.at(i)->getImageView().handle);
	}

	createFramebuffers(*this->window, this->framebuffers, this->renderPass->getRenderPassHandle(), shadowViews, *this->renderExtent);
}