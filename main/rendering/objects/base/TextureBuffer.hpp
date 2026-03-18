#pragma once

#include "../../../vulkan/objects/VkObjects.hpp"
#include "../../../vulkan/objects/VkImage.hpp"
#include "interfaces/ITextureBufferListener.hpp"

#include "structure/Textures.hpp"

struct VulkanContext;

class TextureBuffer {
public:
	TextureBuffer() = default;
	TextureBuffer(VulkanContext* context);

	~TextureBuffer() = default;

	void recreate();

	void addListener(ITextureBufferListener* listener);
	void removeListener(ITextureBufferListener* listener);

	ImageFormat getFormat();

	vk::Image& getImage();
	vk::ImageView& getImageView();
protected:
	VulkanContext* context;

	std::vector<ITextureBufferListener*> listeners;

	vk::Image image;
	vk::ImageView imageView;

	VkFormat format = VK_FORMAT_UNDEFINED;

	VkExtent2D* renderExtent = nullptr;
public:
	class Builder {
	public:
		static Builder* get() { return new Builder(); }

		Builder* withDescription(TextureDesc textureDesc);
		Builder* withExtent(VkExtent2D* extent);

		TextureBuffer build();
	private:
		Builder();

		TextureDesc textureDesc;
		VkExtent2D* extent = nullptr;
	};
};