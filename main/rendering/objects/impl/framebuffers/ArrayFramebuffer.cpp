#include "ArrayFramebuffer.hpp"

#include "../../base/ArrayTextureBuffer.hpp"
#include "../../../PipelineCreation.hpp"
#include "../../../../vulkan/VulkanDevice.hpp"

#include "Error.hpp"
#include "toString.hpp"

ArrayFramebuffer::ArrayFramebuffer(
	VulkanWindow* window,
	std::initializer_list<TextureBuffer*> textureBuffers,
	RenderPass* renderPass,
	std::uint32_t arraySize,
	VkExtent2D* shadowMapResolution
) : arraySize(arraySize),  
	textureBuffers(textureBuffers), 
	renderPass(renderPass), 
	Framebuffer(window) 
{
	this->renderExtent = shadowMapResolution;

	this->recreate();
}

void ArrayFramebuffer::recreate() {
	this->framebuffers.clear();

	// Due to different setup I manaully create the framebuffers instead.
	
	std::vector<VkImageView> imageViews;
	imageViews.resize(this->textureBuffers.size(), VK_NULL_HANDLE);

	VkFramebufferCreateInfo fbInfo = {
		.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
		.renderPass = this->renderPass->getRenderPassHandle(),
		.attachmentCount = static_cast<std::uint32_t>(this->textureBuffers.size()),
		.pAttachments = imageViews.data(),
		.width = this->renderExtent->width,
		.height = this->renderExtent->height,
		.layers = 1
	};

	for (std::uint32_t i = 0; i < this->arraySize; i++) {
		for (std::size_t j = 0; j < this->textureBuffers.size(); j++) {
			imageViews[j] = dynamic_cast<ArrayTextureBuffer*>(this->textureBuffers[j])->getFramebufferViews()[i].handle;
		}

		VkFramebuffer fb = VK_NULL_HANDLE;
		if (const auto res = vkCreateFramebuffer(this->window->device->device, &fbInfo, nullptr, &fb); VK_SUCCESS != res)
			throw Utils::Error("Unable to create framebuffer for swap chain image %zu\n vkCreateFramebuffer() returned %s", i, Utils::toString(res).c_str());

		this->framebuffers.emplace_back(vk::Framebuffer(this->window->device->device, fb));
	}
}