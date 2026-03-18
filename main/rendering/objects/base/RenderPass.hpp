#pragma once

#include <vector>
#include <optional>
#include <memory>

#include "../../../vulkan/objects/VkObjects.hpp"
#include "structure/Attachments.hpp"
#include "structure/Pipelines.hpp"
#include "../impl/Textures.hpp"

class VulkanWindow;

class RenderPass {
public:
	~RenderPass() = default;

	// Delete copy constructors
	RenderPass(const RenderPass& other) = delete;
	RenderPass& operator=(const RenderPass& other) = delete;

	// Define move constructors
	RenderPass(RenderPass&& other) noexcept;
	RenderPass& operator=(RenderPass&& other) noexcept;

	void recreate();

	vk::RenderPass& getRenderPass();
	VkRenderPass getRenderPassHandle();
	std::vector<VkClearValue>& getClearValues();
protected:
	RenderPass() = default;
	RenderPass(VulkanWindow* window);

	VulkanWindow* window;

	vk::RenderPass renderPass;
	std::vector<VkClearValue> clearValues;

public:
	class Builder {
	public:
		static Builder* get() { return new Builder(); }

		Builder* withColourAttachment(Texture colourAttachment, AttachmentLoadOp loadOp, AttachmentStoreOp storeOp, ImageLayout layout);
		Builder* withDepthAttachment(Texture depthAttachment, AttachmentLoadOp loadOp, AttachmentStoreOp storeOp, ImageLayout layout);
		Builder* usesDescriptorInShader(ImageType imageType);

		RenderPass build();
	private:
		Builder();

		std::vector<AttachmentDesc> attachments;
		std::optional<uint32_t> depthTextureIndex;
		std::optional<ImageType> descriptorType = std::nullopt;
	};
};

