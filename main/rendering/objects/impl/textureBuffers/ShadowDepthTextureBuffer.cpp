#include "ShadowDepthTextureBuffer.hpp"

#include "../../../PipelineCreation.hpp"
#include "../../../../vulkan/VulkanContext.hpp"
#include "../../../../vulkan/Swapchain.hpp"

ShadowDepthTextureBuffer::ShadowDepthTextureBuffer(
	VulkanContext* context,
	VkExtent2D* renderExtent) : TextureBuffer(context) 
{
	if (!renderExtent)
		this->renderExtent = &this->context->window->getSwapchain()->getExtent();
	else
		this->renderExtent = renderExtent;

	this->recreate();
}

void ShadowDepthTextureBuffer::recreate() {
	TextureBufferSetting textureSetting = {
		.imageFormat = VK_FORMAT_D32_SFLOAT,
		.imageExtent = *this->renderExtent,
		.imageUsage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
		.viewAspectFlags = VK_IMAGE_ASPECT_DEPTH_BIT };

	std::pair<vk::Image, vk::ImageView> textureBuffer = createTextureBuffer(*this->context, textureSetting);

	this->image = std::move(textureBuffer.first);
	this->imageView = std::move(textureBuffer.second);

	TextureBuffer::recreate();
}