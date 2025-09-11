#include "CubemapArrayFramebuffer.hpp"

#include "../../../PipelineCreation.hpp"

#include "../textureBuffers/CubemapArrayDepthTextureBuffer.hpp"
#include "../../../../vulkan/VulkanDevice.hpp"

#include "Error.hpp"
#include "toString.hpp"

CubemapArrayFramebuffer::CubemapArrayFramebuffer(
	VulkanWindow* window,
	_TextureBuffer* textureBuffer,
	_RenderPass* renderPass,
	std::uint32_t arraySize,
	VkExtent2D* shadowMapResolution) : arraySize(arraySize), Framebuffer(window) 
{
	this->textureBuffer = textureBuffer;
	this->renderPass = renderPass;

	this->renderExtent = shadowMapResolution;

	this->recreate();
}

void CubemapArrayFramebuffer::recreate() {
	this->framebuffers.clear();

	// Due to different setup I manaully create the framebuffers instead.

	VkImageView imageView[1]{};

	VkFramebufferCreateInfo fbInfo{};
	fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
	fbInfo.renderPass = this->renderPass->get()->getRenderPassHandle();
	fbInfo.attachmentCount = 1;
	fbInfo.pAttachments = imageView;
	fbInfo.width = this->renderExtent->width;
	fbInfo.height = this->renderExtent->height;
	fbInfo.layers = 1;

	for (std::size_t i = 0; i < 6 * this->arraySize; i++) {
		imageView[0] = dynamic_cast<CubemapArrayDepthTextureBuffer*>(this->textureBuffer->get())->getFramebufferViews()[i].handle;

		VkFramebuffer fb = VK_NULL_HANDLE;
		if (const auto res = vkCreateFramebuffer(this->window->device->device, &fbInfo, nullptr, &fb); VK_SUCCESS != res)
			throw Utils::Error("Unable to create framebuffer for swap chain image %zu\n vkCreateFramebuffer() returned %s", i, Utils::toString(res).c_str());

		// NOTE: Similarly to CubemapFrambuffer.cpp, this must not be indexed via the imageIndex, rather indexed by the face being rendered to.
		// Specifically, indexed via the calculation (arrayIndex * 6) + face.
		this->framebuffers.emplace_back(vk::Framebuffer(this->window->device->device, fb));
	}
}