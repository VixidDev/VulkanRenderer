#include "CubemapFramebuffer.hpp"

#include "../../../PipelineCreation.hpp"

#include "../textureBuffers/CubemapDepthTextureBuffer.hpp"
#include "../../../../vulkan/VulkanDevice.hpp"

#include "Error.hpp"
#include "toString.hpp"

CubemapFramebuffer::CubemapFramebuffer(
	VulkanWindow* window,
	TextureBuffer* textureBuffer,
	RenderPass* renderPass,
	VkExtent2D* shadowMapResolution) : Framebuffer(window) {
	this->textureBuffer = textureBuffer;
	this->renderPass = renderPass;

	this->renderExtent = shadowMapResolution;

	this->recreate();
}

void CubemapFramebuffer::recreate() {
	this->framebuffers.clear();

	// Due to different setup I manaully create the framebuffers instead.

	VkImageView imageView[1]{};

	VkFramebufferCreateInfo fbInfo{};
	fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
	fbInfo.renderPass = this->renderPass->getRenderPassHandle();
	fbInfo.attachmentCount = 1;
	fbInfo.pAttachments = imageView;
	fbInfo.width = this->renderExtent->width;
	fbInfo.height = this->renderExtent->height;
	fbInfo.layers = 1;

	for (std::size_t i = 0; i < 6; i++) {
		imageView[0] = dynamic_cast<CubemapDepthTextureBuffer*>(this->textureBuffer)->getFramebufferViews()[i].handle;

		VkFramebuffer fb = VK_NULL_HANDLE;
		if (const auto res = vkCreateFramebuffer(this->window->device->device, &fbInfo, nullptr, &fb); VK_SUCCESS != res)
			throw Utils::Error("Unable to create framebuffer for swap chain image %zu\n vkCreateFramebuffer() returned %s", i, Utils::toString(res).c_str());
		
		// NOTE: This must not be indexed via the imageIndex, rather indexed by the face being rendered to.
		// While this is counter-intuitive to how framebuffers are indexed in the Renderer.cpp I elected to do this
		// to avoid complicating specific objects and requiring downcasting just to correctly use the object.
		this->framebuffers.emplace_back(vk::Framebuffer(this->window->device->device, fb));
	}
}