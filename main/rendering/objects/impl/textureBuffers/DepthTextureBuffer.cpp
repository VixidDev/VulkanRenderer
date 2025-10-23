#include "DepthTextureBuffer.hpp"

#include "../../../PipelineCreation.hpp"
#include "../../../../vulkan/VulkanContext.hpp"
#include "../../../../vulkan/Swapchain.hpp"

// Since this is almost identical to ColourTextureBuffer I could probably
// merge them and infer the imageUsage and aspectFlags based on the given format
DepthTextureBuffer::DepthTextureBuffer(
	VulkanContext* context,
	VkFormat format,
	VkExtent2D* renderExtent
) : TextureBuffer(context)
{
	this->format = format;
	
	if (!renderExtent)
		this->renderExtent = &this->context->window->getSwapchain()->getExtent();
	else
		this->renderExtent = renderExtent;

	this->recreate();
}

void DepthTextureBuffer::recreate() {
	TextureBufferSetting textureSetting = {
		.imageFormat = this->format,
		.imageExtent = *this->renderExtent,
		.imageUsage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
		.viewAspectFlags = VK_IMAGE_ASPECT_DEPTH_BIT,
		.samples = VK_SAMPLE_COUNT_1_BIT };

	std::pair<vk::Image, vk::ImageView> textureBuffer = createTextureBuffer(*this->context, textureSetting);

	this->image = std::move(textureBuffer.first);
	this->imageView = std::move(textureBuffer.second);

	TextureBuffer::recreate();
}