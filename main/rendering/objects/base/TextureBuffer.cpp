#include "TextureBuffer.hpp"

#include "../vulkan/Swapchain.hpp"
#include "../vulkan/VulkanDevice.hpp"
#include "error.hpp"
#include "toString.hpp"

TextureBuffer::TextureBuffer(
	VkImageCreateInfo imageCreateInfo, 
	VkImageViewCreateInfo imageViewCreateInfo, 
	ImageFormat format,
	ExtentRatio extentRatio,
	TextureUseFlags futureUse,
	bool isRenderTarget,
	bool calcMipmaps
) : imageCreateInfo(imageCreateInfo), 
	imageViewCreateInfo(imageViewCreateInfo),
	format(format),
	extentRatio(extentRatio),
	futureUse(futureUse),
	isRenderTarget(isRenderTarget),
	useMipmaps(calcMipmaps) {}

VkImage TextureBuffer::getImage(std::shared_ptr<VulkanContext> context) {
	if (this->image.has_value()) return this->image.value().image;

	this->compile(context);

	return this->image.value().image;
}

VkImageView TextureBuffer::getImageView(std::shared_ptr<VulkanContext> context) {
	if (this->imageView.has_value()) return this->imageView.value().handle;

	this->compile(context);

	return this->imageView.value().handle;
}

void TextureBuffer::compile(std::shared_ptr<VulkanContext> context) {
	if (!this->context) this->context = context;

	// imageCreateInfo is incomplete at this point, need to use context
	// to gather information to complete it and build the Vulkan objects

	Swapchain* swapchain = this->context->window->getSwapchain();
	// Get function handle to get specified extent
	if (!this->extentFunc) {
		switch (this->extentRatio) {
		case ExtentRatio::SWAPCHAIN:
			this->extentFunc = [&swapchain]() -> VkExtent2D { swapchain->getExtent(); };
			break;
		case ExtentRatio::HALF_SWAPCHAIN:
			this->extentFunc = [&swapchain]() -> VkExtent2D { swapchain->getHalfExtent(); };
			break;
		case ExtentRatio::QUARTER_SWAPCHAIN:
			this->extentFunc = [&swapchain]() -> VkExtent2D { swapchain->getQuarterExtent(); };
			break;
		}
	}

	// Invoke extent func to get actual extent
	VkExtent2D extent = this->extentFunc();

	// Get mipmap level if needed
	std::uint32_t mipLevels = 1;
	if (this->useMipmaps)
		mipLevels = computeMipLevels(extent.width, extent.height);

	// Fill missing info in imageCreateInfo
	this->imageCreateInfo.extent.width = extent.width;
	this->imageCreateInfo.extent.height = extent.height;
	this->imageCreateInfo.mipLevels = mipLevels;

	// Define VMA allocation info
	VmaAllocationCreateInfo allocInfo{};
	allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
	allocInfo.requiredFlags = 0;
	allocInfo.preferredFlags = 0;

	VulkanAllocator* allocator = this->context->allocator.get();

	VkImage image = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;
	if (const VkResult res = vmaCreateImage(allocator->allocator, &this->imageCreateInfo, &allocInfo, &image, &allocation, nullptr); VK_SUCCESS != res)
		throw Utils::Error("Unable to allocate TextureBuffer.\nvmaCreateImage() returned %s\n", Utils::toString(res).c_str());

	vk::Image Image(allocator->allocator, image, allocation);

	// Fill view create info
	this->imageViewCreateInfo.image = image;

	VulkanDevice* device = this->context->window->getDevice();

	VkImageView view = VK_NULL_HANDLE;
	if (const VkResult res = vkCreateImageView(device->getDevice(), &this->imageViewCreateInfo, nullptr, &view); VK_SUCCESS != res)
		throw Utils::Error("Unable to create image view.\nvkCreateImageView() returned %s\n", Utils::toString(res).c_str());

	this->image = std::move(Image);
	this->imageView = vk::ImageView(device->getDevice(), view);

	// If we don't render to this texture buffer return now
	if (!this->isRenderTarget) return;

	// Else set up views to render into, taking into consideration
	// array layers and if texture is a cubemap

	// Change view info to single layer, 2D view type for framebuffer views.
	// Get a copy of the class member image create info as to not change
	// the original info struct
	VkImageViewCreateInfo viewInfo = this->imageViewCreateInfo;
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	viewInfo.subresourceRange.layerCount = 1;

	this->framebufferViews.clear();

	std::uint32_t arrayLayers = this->imageCreateInfo.arrayLayers;

	if (this->imageCreateInfo.usage & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) {
		// Arrays are multiples of 6 when it is a cubemap
		std::uint32_t elements = arrayLayers / 6;
		for (std::uint32_t element = 0; element < elements; element++) {
			for (std::uint32_t face = 0; face < 6; face++) {
				std::uint32_t layer = (element * 6) + face;

				viewInfo.subresourceRange.baseArrayLayer = layer;

				VkImageView fbView = VK_NULL_HANDLE;
				if (const VkResult res = vkCreateImageView(device->getDevice(), &viewInfo, nullptr, &fbView); VK_SUCCESS != res)
					throw Utils::Error("Unable to create image view.\nvkCreateImageView() returned %s\n", Utils::toString(res).c_str());

				this->framebufferViews.emplace_back(vk::ImageView(device->getDevice(), fbView));
			}
		}
	} else {
		for (std::uint32_t element = 0; element < arrayLayers; element++) {
			viewInfo.subresourceRange.baseArrayLayer = element;

			VkImageView fbView = VK_NULL_HANDLE;
			if (const VkResult res = vkCreateImageView(device->getDevice(), &viewInfo, nullptr, &fbView); VK_SUCCESS != res)
				throw Utils::Error("Unable to create image view.\nvkCreateImageView() returned %s\n", Utils::toString(res).c_str());

			this->framebufferViews.emplace_back(vk::ImageView(device->getDevice(), fbView));
		}
	}
}

void TextureBuffer::recreate() {
	for (ITextureBufferListener* listener : this->listeners) {
		if (!listener) continue;
		listener->onTextureBufferRecreated();
	}
}

void TextureBuffer::addListener(ITextureBufferListener* listener) {
	this->listeners.push_back(listener);
}

void TextureBuffer::removeListener(ITextureBufferListener* listener) {
	this->listeners.erase(std::remove(this->listeners.begin(), this->listeners.end(), listener), this->listeners.end());
}

TextureBuffer::Builder* TextureBuffer::Builder::withDescription(TextureDesc textureDesc) {
	this->textureDesc = textureDesc;
	return this;
}

TextureBuffer::Builder* TextureBuffer::Builder::withFlags(ImageCreateFlags imageCreateFlags) {
	this->createFlags = imageCreateFlags;
	return this;
}

TextureBuffer::Builder* TextureBuffer::Builder::withExtent(ExtentRatio extentRatio) {
	this->extent = extent;
	return this;
}

TextureBuffer::Builder* TextureBuffer::Builder::withArrayLayers(std::uint32_t arrayLayers) {
	this->arrayLayers = arrayLayers;
	return this;
}

TextureBuffer::Builder* TextureBuffer::Builder::withSamples(ImageSamples samples) {
	this->samples = samples;
	return this;
}

TextureBuffer::Builder* TextureBuffer::Builder::withViewType(ImageViewType viewType) {
	this->imageViewType = viewType;
	return this;
}

TextureBuffer::Builder* TextureBuffer::Builder::hasFutureUse(TextureUseFlags futureUse) {
	this->futureUse = futureUse;
	return this;
}

TextureBuffer::Builder* TextureBuffer::Builder::isRenderTarget() {
	this->shouldRenderTo = true;
	return this;
}

TextureBuffer::Builder* TextureBuffer::Builder::useMipmaps(bool value) {
	this->calcMipmaps = value;
	return this;
}

TextureBuffer TextureBuffer::Builder::build() {
	std::uint32_t arrayLayers = this->createFlags & ImageCreate::CUBE_COMPATIBLE ? 6 * this->arrayLayers : this->arrayLayers;
	VkImageUsageFlags usage = this->shouldRenderTo ? this->textureDesc.usage | ImageUsage::TRANSFER_DST : this->textureDesc.usage;

	VkImageCreateInfo imageInfo{};
	imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageInfo.flags = this->createFlags;
	imageInfo.imageType = VK_IMAGE_TYPE_2D;
	imageInfo.format = (VkFormat)this->textureDesc.format;
	//imageInfo.extent.width;
	//imageInfo.extent.height;
	imageInfo.extent.depth = 1;
	//imageInfo.mipLevels;
	imageInfo.arrayLayers = arrayLayers;
	imageInfo.samples = (VkSampleCountFlagBits)this->samples;
	imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageInfo.usage = usage;
	imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

	VkImageViewCreateInfo viewInfo{};
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	//viewInfo.image;
	viewInfo.viewType = (VkImageViewType)this->imageViewType;
	viewInfo.format = (VkFormat)this->textureDesc.format;
	viewInfo.components = VkComponentMapping{};
	viewInfo.subresourceRange = VkImageSubresourceRange{
		(VkImageAspectFlags)this->textureDesc.aspect,
		0, // baseMipLevel
		1, // levelCount
		0, // baseArrayLayer
		arrayLayers // layerCount
	};

	return TextureBuffer(
		imageInfo, 
		viewInfo, 
		this->textureDesc.format, 
		this->extent, 
		this->futureUse, 
		this->shouldRenderTo, 
		this->calcMipmaps
	);
}

std::uint32_t computeMipLevels(std::uint32_t width, std::uint32_t height) {
	const std::uint32_t bits = width | height;
	const std::uint32_t leadingZeros = std::countl_zero(bits);
	return 32 - leadingZeros;
}