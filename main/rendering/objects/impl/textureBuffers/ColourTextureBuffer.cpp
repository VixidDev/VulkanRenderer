#include "ColourTextureBuffer.hpp"

#include "../../../PipelineCreation.hpp"
#include "../../../../vulkan/VulkanContext.hpp"
#include "../../../../vulkan/Swapchain.hpp"

ColourTextureBuffer::ColourTextureBuffer(
	VulkanContext* context,
	VkFormat format,
	VkExtent2D* renderExtent
) : TextureBuffer(context) {
	this->format = format;

	if (!renderExtent)
		this->renderExtent = &this->context->window->getSwapchain()->getExtent();
	else
		this->renderExtent = renderExtent;

	this->recreate();
}

void ColourTextureBuffer::recreate() {
	TextureBufferSetting textureSetting = {
		.imageFormat = this->format,
		.imageExtent = *this->renderExtent,
		.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
		.viewAspectFlags = VK_IMAGE_ASPECT_COLOR_BIT,
		.samples = VK_SAMPLE_COUNT_1_BIT };

	std::pair<vk::Image, vk::ImageView> textureBuffer = createTextureBuffer(*this->context, textureSetting);

	this->image = std::move(textureBuffer.first);
	this->imageView = std::move(textureBuffer.second);

	TextureBuffer::recreate();
}