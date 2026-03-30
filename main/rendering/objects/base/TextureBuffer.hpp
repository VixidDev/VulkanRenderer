#pragma once

#include "../../../vulkan/objects/VkObjects.hpp"
#include "../../../vulkan/objects/VkImage.hpp"
#include "interfaces/ITextureBufferListener.hpp"

#include "structure/Textures.hpp"

#include <optional>
#include <functional>

struct VulkanContext;

class TextureBuffer {
public:
	~TextureBuffer() = default;

	// Delete copy constructors
	TextureBuffer(const TextureBuffer& other) = delete;
	TextureBuffer& operator=(const TextureBuffer& other) = delete;

	// Define move constructors
	TextureBuffer(TextureBuffer&& other) noexcept;
	TextureBuffer& operator=(TextureBuffer&& other) noexcept;

	VkImage getImage(std::shared_ptr<VulkanContext> context);
	VkImageView getImageView(std::shared_ptr<VulkanContext> context);

	void recreate();

	void addListener(ITextureBufferListener* listener);
	void removeListener(ITextureBufferListener* listener);

	ImageFormat getFormat() const { return format; }
	TextureUseFlags getFutureUse() const { return futureUse; }
protected:
	TextureBuffer() = default;
	TextureBuffer(
		VkImageCreateInfo imageCreateInfo, 
		VkImageViewCreateInfo imageViewCreateInfo, 
		ImageFormat format,
		ExtentRatio extentRatio,
		TextureUseFlags futureUse,
		bool isRenderTarget,
		bool calcMipmaps);

	void compile(std::shared_ptr<VulkanContext> context);

	VkImageCreateInfo imageCreateInfo;
	VkImageViewCreateInfo imageViewCreateInfo;
	ImageFormat format;
	ExtentRatio extentRatio;
	TextureUseFlags futureUse;
	bool isRenderTarget;
	bool useMipmaps;

	std::shared_ptr<VulkanContext> context;

	std::function<VkExtent2D()> extentFunc = nullptr;

	std::optional<vk::Image> image = std::nullopt;
	std::optional<vk::ImageView> imageView = std::nullopt;
	std::vector<vk::ImageView> framebufferViews;

	std::vector<ITextureBufferListener*> listeners;
public:
	class Builder {
	public:
		static Builder* get() { return new Builder(); }

		Builder* withDescription(TextureDesc textureDesc);
		Builder* withFlags(ImageCreateFlags imageCreateFlags);
		Builder* withExtent(ExtentRatio extentRatio);
		Builder* withArrayLayers(std::uint32_t arrayLayers);
		Builder* withSamples(ImageSamples samples);
		Builder* withViewType(ImageViewType viewType);
		Builder* hasFutureUse(TextureUseFlags futureUse);
		Builder* isRenderTarget();
		Builder* useMipmaps(bool value = false);

		TextureBuffer build();
	private:
		Builder();

		TextureDesc textureDesc;
		ExtentRatio extent = ExtentRatio::SWAPCHAIN;
		ImageCreateFlags createFlags = ImageCreate::NONE;
		std::uint32_t arrayLayers = 1;
		ImageSamples samples = ImageSamples::ONE;
		ImageViewType imageViewType = ImageViewType::TYPE_2D;
		TextureUseFlags futureUse = TextureUse::NONE;
		bool shouldRenderTo = false;
		bool calcMipmaps = false;
	};
};