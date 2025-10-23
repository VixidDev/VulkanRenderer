#pragma once

#include "../../../vulkan/objects/VkObjects.hpp"
#include "../../../vulkan/objects/VkImage.hpp"
#include "interfaces/ITextureBufferListener.hpp"

struct VulkanContext;

class TextureBuffer {
public:
	TextureBuffer() = default;
	TextureBuffer(VulkanContext* context);

	virtual ~TextureBuffer() = default;

	virtual void recreate();

	void addListener(ITextureBufferListener* listener);
	void removeListener(ITextureBufferListener* listener);

	vk::Image& getImage();
	virtual vk::ImageView& getImageView();
protected:
	VulkanContext* context;

	std::vector<ITextureBufferListener*> listeners;

	vk::Image image;
	vk::ImageView imageView;

	VkFormat format = VK_FORMAT_UNDEFINED;

	VkExtent2D* renderExtent = nullptr;
};