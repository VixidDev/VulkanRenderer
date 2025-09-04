#include "DepthTextureBuffer.hpp"

#include "../../../../vulkan/VulkanContext.hpp"
#include "../../../PipelineCreation.hpp"

DepthTextureBuffer::DepthTextureBuffer(
	VulkanContext* context,
	VkSampleCountFlagBits* sampleCount, 
	VkExtent2D* renderExtent) : TextureBuffer(context)
{
	this->sampleCount = sampleCount;
	
	if (!renderExtent)
		this->renderExtent = &this->context->window->swapchainExtent;
	else
		this->renderExtent = renderExtent;

	this->recreate();
}

void DepthTextureBuffer::recreate() {
	TextureBufferSetting textureSetting = {
		.imageFormat = VK_FORMAT_D32_SFLOAT,
		.imageExtent = *this->renderExtent,
		.imageUsage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT,
		.viewAspectFlags = VK_IMAGE_ASPECT_DEPTH_BIT };

	std::pair<vk::Image, vk::ImageView> textureBuffer = createTextureBuffer(*this->context, textureSetting);

	this->image = std::move(textureBuffer.first);
	this->imageView = std::move(textureBuffer.second);

	TextureBuffer::recreate();
}