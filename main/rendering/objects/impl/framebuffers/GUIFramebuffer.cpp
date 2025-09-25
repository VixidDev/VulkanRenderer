#include "GUIFramebuffer.hpp"

#include "../../../../vulkan/VulkanWindow.hpp"
#include "../../../../vulkan/VulkanDevice.hpp"

#include "Error.hpp"
#include "toString.hpp"

GUIFramebuffer::GUIFramebuffer(VulkanWindow* window, RenderPass* renderPass) : Framebuffer(window) {
	this->renderPass = renderPass;

	this->renderExtent = &this->window->swapchainExtent;

	this->recreate();
}

void GUIFramebuffer::recreate() {
	this->framebuffers.clear();

	for (std::size_t i = 0; i < this->window->swapViews.size(); ++i) {
		VkFramebufferCreateInfo fbInfo{};
		fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		fbInfo.flags = 0;
		fbInfo.renderPass = this->renderPass->getRenderPassHandle();
		fbInfo.attachmentCount = 1;
		fbInfo.pAttachments = &this->window->swapViews.at(i);
		fbInfo.width = this->renderExtent->width;
		fbInfo.height = this->renderExtent->height;
		fbInfo.layers = 1;

		VkFramebuffer fb = VK_NULL_HANDLE;
		if (const auto res = vkCreateFramebuffer(this->window->device->device, &fbInfo, nullptr, &fb); VK_SUCCESS != res)
			throw Utils::Error("Unable to create framebuffer for swap chain image %zu\n vkCreateFramebuffer() returned %s", i, Utils::toString(res).c_str());

		this->framebuffers.emplace_back(vk::Framebuffer(this->window->device->device, fb));
	}
}