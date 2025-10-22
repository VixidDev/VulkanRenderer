#include "CubemapArrayFramebuffer.hpp"

#include "../../../PipelineCreation.hpp"

#include "../../base/ArrayTextureBuffer.hpp"
#include "../../../../vulkan/VulkanDevice.hpp"

#include "Error.hpp"
#include "toString.hpp"

CubemapArrayFramebuffer::CubemapArrayFramebuffer(
	VulkanWindow* window,
	TextureBuffer* textureBuffer,
	RenderPass* renderPass,
	std::uint32_t arraySize,
	VkExtent2D* renderExtent) : arraySize(arraySize), Framebuffer(window) 
{
	this->textureBuffer = textureBuffer;
	this->renderPass = renderPass;

	this->renderExtent = renderExtent;

	this->recreate();
}

void CubemapArrayFramebuffer::recreate() {
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

	for (std::size_t i = 0; i < 6 * this->arraySize; i++) {
		// This line does make it so this class always depends on using a CubemapArrayDepthTextureBuffer, this should probably be changed
		// to be a type that is templated on this class and not hardcoded.
		imageView[0] = dynamic_cast<ArrayTextureBuffer*>(this->textureBuffer)->getFramebufferViews()[i].handle;

		VkFramebuffer fb = VK_NULL_HANDLE;
		if (const auto res = vkCreateFramebuffer(this->window->getDevice()->getDevice(), &fbInfo, nullptr, &fb); VK_SUCCESS != res)
			throw Utils::Error("Unable to create framebuffer for swap chain image %zu\n vkCreateFramebuffer() returned %s", i, Utils::toString(res).c_str());

		// NOTE: Similarly to CubemapFrambuffer.cpp, this must not be indexed via the imageIndex, rather indexed by the face being rendered to.
		// Specifically, indexed via the calculation (arrayIndex * 6) + face.
		this->framebuffers.emplace_back(vk::Framebuffer(this->window->getDevice()->getDevice(), fb));
	}
}