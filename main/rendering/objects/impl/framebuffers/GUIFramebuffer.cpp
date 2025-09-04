#include "GUIFramebuffer.hpp"

#include "../../../PipelineCreation.hpp"

GUIFramebuffer::GUIFramebuffer(VulkanWindow* window, _RenderPass* renderPass) : Framebuffer(window) {
	this->renderPass = renderPass;

	this->renderExtent = &this->window->swapchainExtent;

	this->recreate();
}

void GUIFramebuffer::recreate() {
	this->framebuffers.clear();

	std::vector<VkImageView> guiViews;

	createFramebuffers(*this->window, this->framebuffers, this->renderPass->get()->getRenderPassHandle(), guiViews, *this->renderExtent);
}