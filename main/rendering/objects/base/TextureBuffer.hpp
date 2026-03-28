#pragma once

#include "../../../vulkan/objects/VkObjects.hpp"
#include "../../../vulkan/objects/VkImage.hpp"
#include "interfaces/ITextureBufferListener.hpp"

#include "structure/Textures.hpp"

#include <optional>

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

	void recreate();

	void addListener(ITextureBufferListener* listener);
	void removeListener(ITextureBufferListener* listener);

	ImageFormat getFormat() const { return format; }
	TextureUseFlags getFutureUse() const { return futureUse; }

	vk::Image& getImage();
	vk::ImageView& getImageView();
protected:
	TextureBuffer() = default;
	TextureBuffer(VulkanContext* context);

	VulkanContext* context;

	std::vector<ITextureBufferListener*> listeners;

	vk::Image image;
	vk::ImageView imageView;

	VkFormat _format = VK_FORMAT_UNDEFINED;
	ImageFormat format;
	TextureUseFlags futureUse = TextureUse::NONE;

	VkExtent2D* renderExtent = nullptr;
public:
	class Builder {
	public:
		static Builder* get() { return new Builder(); }

		Builder* withDescription(TextureDesc textureDesc);
		Builder* withExtent(ExtentRatio extent);
		Builder* hasFutureUse(TextureUseFlags futureUse);

		TextureBuffer build();
	private:
		Builder();

		TextureDesc textureDesc;
		ExtentRatio extent = ExtentRatio::SWAPCHAIN;
		TextureUseFlags futureUse = TextureUse::NONE;
	};
};