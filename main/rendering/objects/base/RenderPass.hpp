#pragma once

#include "structure/Attachments.hpp"
#include "structure/Pipelines.hpp"
#include "../impl/Textures.hpp"
#include "../vulkan/VulkanDevice.hpp"

#include <vector>
#include <memory>
#include <array>

class VulkanDevice;

class RenderPass {
public:
	~RenderPass() = default;

	// Delete copy constructors
	RenderPass(const RenderPass& other) = delete;
	RenderPass& operator=(const RenderPass& other) = delete;

	// Define move constructors
	RenderPass(RenderPass&& other) noexcept;
	RenderPass& operator=(RenderPass&& other) noexcept;

	VkRenderPass get(std::shared_ptr<VulkanDevice> device);

	void recreate();

	vk::RenderPass& getRenderPass();
private:
	RenderPass() = default;
	RenderPass(
		const std::vector<VkAttachmentDescription> attachmentDescriptions,
		const std::vector<VkAttachmentReference> attachmentReferences,
		VkSubpassDescription subpassDescription,
		const std::array<VkSubpassDependency, 2> subpassDependencies
	);

	VkRenderPass compile(std::shared_ptr<VulkanDevice> device);

	std::vector<VkAttachmentDescription> attachmentDescriptions;
	std::vector<VkAttachmentReference> attachmentReferences;
	VkSubpassDescription subpassDescription;
	std::array<VkSubpassDependency, 2> subpassDependencies;

	std::shared_ptr<VulkanDevice> device;

	std::optional<vk::RenderPass> renderPass = std::nullopt;
public:
	class Builder {
	public:
		static Builder* get() { return new Builder(); }

		Builder* withColourAttachment(
			Texture colourAttachment, 
			LoadOp loadOp, 
			StoreOp storeOp, 
			ImageLayout finalLayout, 
			ImageLayout initialLayout = ImageLayout::UNDEFINED
		);
		Builder* withDepthAttachment(
			Texture depthAttachment, 
			LoadOp loadOp, 
			StoreOp storeOp, 
			ImageLayout finalLayout, 
			ImageLayout initialLayout = ImageLayout::UNDEFINED
		);
		Builder* usesDescriptorInShader(ImageTypeFlags imageType);

		RenderPass build();
	private:
		Builder();

		std::vector<AttachmentDesc> attachments;
		std::optional<uint32_t> depthTextureIndex = std::nullopt;
		std::optional<ImageTypeFlags> descriptorType = std::nullopt;
	};
};

