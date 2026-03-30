#include "RenderPass.hpp"

#include "error.hpp"
#include "toString.hpp"
#include "../../../vulkan/VkUtils.hpp"

RenderPass::RenderPass(
	const std::vector<VkAttachmentDescription> attachmentDescriptions,
	const std::vector<VkAttachmentReference> attachmentReferences,
	VkSubpassDescription subpassDescription,
	const std::array<VkSubpassDependency, 2> subpassDependencies
) : attachmentDescriptions(attachmentDescriptions),
	attachmentReferences(attachmentReferences),
	subpassDescription(subpassDescription),
	subpassDependencies(subpassDependencies) {}

VkRenderPass RenderPass::get(std::shared_ptr<VulkanDevice> device) {
	if (this->renderPass.has_value()) return this->renderPass.value().handle;

	return this->compile(device);
}

void RenderPass::recreate() {}

vk::RenderPass& RenderPass::getRenderPass() {
	return this->renderPass.value();
}

VkRenderPass RenderPass::compile(std::shared_ptr<VulkanDevice> device) {
	// Share the pointer if not already
	if (!this->device) this->device = device;

	VkRenderPassCreateInfo passInfo = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
		.attachmentCount = this->attachmentDescriptions.size(),
		.pAttachments = this->attachmentDescriptions.data(),
		.subpassCount = 1,
		.pSubpasses = &this->subpassDescription,
		.dependencyCount = 2,
		.pDependencies = this->subpassDependencies.data()
	};

	VkRenderPass renderPass = VK_NULL_HANDLE;
	if (const VkResult res = vkCreateRenderPass(this->device->getDevice(), &passInfo, nullptr, &renderPass); VK_SUCCESS != res) {
		throw Utils::Error("Unable to create render pass\nvkCreateRenderPass() returned '%s'\n", Utils::toString(res).c_str());
	}

	this->renderPass = vk::RenderPass(this->device->getDevice(), renderPass);
}

/*
 * Render Pass Builder
 */
RenderPass::Builder* RenderPass::Builder::withColourAttachment(
	Texture colourAttachment, LoadOp loadOp, StoreOp storeOp, ImageLayout finalLayout, ImageLayout initialLayout
) {
	AttachmentDesc desc = { colourAttachment, loadOp, storeOp, initialLayout, finalLayout };

	this->attachments.emplace_back(desc);
	return this;
}

RenderPass::Builder* RenderPass::Builder::withDepthAttachment(
	Texture depthAttachment, LoadOp loadOp, StoreOp storeOp, ImageLayout finalLayout, ImageLayout initialLayout
) {
	if (this->depthTextureIndex.has_value())
		throw Utils::Error("Render passes can only have 1 depth buffer bound!\n");
	
	AttachmentDesc desc = { depthAttachment, loadOp, storeOp, initialLayout, finalLayout };

	this->attachments.emplace_back(desc);
	this->depthTextureIndex = this->attachments.size() - 1;
	return this;
}

RenderPass::Builder* RenderPass::Builder::usesDescriptorInShader(ImageTypeFlags imageType) {
	this->descriptorType = imageType;
	return this;
}

RenderPass RenderPass::Builder::build() {
	// Attachment descriptions
	std::vector<VkAttachmentDescription> attachments;
	attachments.reserve(this->attachments.size());

	for (size_t i = 0; i < this->attachments.size(); i++) {
		AttachmentDesc& desc = this->attachments.at(i);
		TextureBuffer& texture = Textures::get(desc.texture);

		VkAttachmentDescription attachmentDesc{};
		attachmentDesc.format = (VkFormat)texture.getFormat();
		attachmentDesc.loadOp = (VkAttachmentLoadOp)desc.loadOp;
		attachmentDesc.storeOp = (VkAttachmentStoreOp)desc.storeOp;
		attachmentDesc.initialLayout = (VkImageLayout)desc.initialLayout;
		attachmentDesc.finalLayout = (VkImageLayout)desc.finalLayout;

		attachments.emplace_back(attachmentDesc);
	}

	// Attachment references
	std::vector<VkAttachmentReference> references;
	references.reserve(this->attachments.size());

	for (size_t i = 0; i < this->attachments.size(); i++) {
		AttachmentDesc& desc = this->attachments.at(i);

		VkAttachmentReference reference{};
		reference.attachment = i;
		reference.layout = (VkImageLayout)desc.finalLayout;

		references.emplace_back(reference);
	}

	VkSubpassDescription subpass{};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = this->attachments.size() - (this->depthTextureIndex ? 1 : 0);
	subpass.pColorAttachments = references.data();
	subpass.pDepthStencilAttachment = this->depthTextureIndex ? &references.at(this->depthTextureIndex.value()) : nullptr;

	// Determine stage and access masks for subpass dependencies

	// Initial subpass dependency masks
	PipelineStageFlags srcStageMask0 = PipelineStage::NONE;
	PipelineStageFlags dstStageMask0 = PipelineStage::NONE;
	AccessFlags srcAccessMask0 = AccessBit::NONE;
	AccessFlags dstAccessMask0 = AccessBit::NONE;

	// Subsequent subpass dependency masks
	PipelineStageFlags srcStageMask1 = PipelineStage::NONE;
	PipelineStageFlags dstStageMask1 = PipelineStage::NONE;
	AccessFlags srcAccessMask1 = AccessBit::NONE;
	AccessFlags dstAccessMask1 = AccessBit::NONE;

	for (size_t i = 0; i < this->attachments.size(); i++) {
		AttachmentDesc& desc = this->attachments.at(i);
		TextureBuffer& texture = Textures::get(desc.texture);
		ImageFormat format = texture.getFormat();

		// If we have attachments with a loadOp of CLEAR, we want to synchronise
		// on depth/color write, since CLEARs are considered write ops
		if (desc.loadOp == LoadOp::CLEAR) {
			if (Textures::isOfDepthFormat(format)) {
				dstStageMask0 |= PipelineStage::EARLY_FRAGMENT;
				dstAccessMask0 |= AccessBit::DEPTH_STENCIL_WRITE;
			} else {
				dstStageMask0 |= PipelineStage::COLOR_OUTPUT;
				dstAccessMask0 |= AccessBit::COLOR_WRITE;
			}
		}

		// If we have attachments with a loadOp of LOAD, we want to synchronise 
		// on depth/color load for that attachment type
		if (desc.loadOp == LoadOp::LOAD) {
			if (Textures::isOfDepthFormat(format)) {
				srcStageMask0 |= PipelineStage::LATE_FRAGMENT;
				srcAccessMask0 |= AccessBit::DEPTH_STENCIL_WRITE;
				dstStageMask0 |= PipelineStage::EARLY_FRAGMENT;
				dstAccessMask0 |= AccessBit::DEPTH_STENCIL_READ;
			} else {
				srcStageMask0 |= PipelineStage::COLOR_OUTPUT;
				srcAccessMask0 |= AccessBit::COLOR_WRITE;
				dstStageMask0 |= PipelineStage::COLOR_OUTPUT;
				dstAccessMask0 |= AccessBit::COLOR_WRITE;
			}
		}

		// If we transition to a differing layout, we need to synchronise the transition
		if (desc.initialLayout != desc.finalLayout) {
			if (Textures::isOfDepthLayout(desc.finalLayout)) {
				srcStageMask0 |= PipelineStage::EARLY_FRAGMENT | PipelineStage::LATE_FRAGMENT;
				srcAccessMask0 |= AccessBit::DEPTH_STENCIL_WRITE;
			} else if (Textures::isOfColorLayout(desc.finalLayout)) {
				srcStageMask0 |= PipelineStage::COLOR_OUTPUT;
				srcAccessMask0 |= AccessBit::COLOR_WRITE;
			}
		}

		// We need to synchronise on any work done this render pass
		if (Textures::isOfDepthFormat(format)) {
			dstStageMask0 |= PipelineStage::EARLY_FRAGMENT;
			// It would be more correct to make the DEPTH_STENCIL dstAccessMasks present depending on the 
			// pipeline depth write/test flags, but it would be more compilcated since multiple pipelines 
			// can use the same render pass, and I don't want to make the same pass with just differing access masks
			dstAccessMask0 |= AccessBit::DEPTH_STENCIL_WRITE | AccessBit::DEPTH_STENCIL_READ;

			srcStageMask1 |= PipelineStage::LATE_FRAGMENT;
			srcAccessMask1 |= AccessBit::DEPTH_STENCIL_WRITE;
		} else {
			dstStageMask0 |= PipelineStage::COLOR_OUTPUT;
			dstAccessMask0 |= AccessBit::COLOR_WRITE;

			srcStageMask1 |= PipelineStage::COLOR_OUTPUT;
			srcAccessMask1 |= AccessBit::COLOR_WRITE;
		}

		// If the attachment has a future use, i.e. is loaded as an attachment in a later pass
		// or is sampled in a shader, we need to synchronise on that too
		if (TextureUseFlags futureUse = texture.getFutureUse()) {
			if (futureUse & TextureUse::ATTACHMENT_LOAD) {
				if (Textures::isOfDepthFormat(format)) {
					dstStageMask1 |= PipelineStage::EARLY_FRAGMENT;
					dstAccessMask1 |= AccessBit::DEPTH_STENCIL_READ;
				} else {
					dstStageMask1 |= PipelineStage::COLOR_OUTPUT;
					dstAccessMask1 |= AccessBit::COLOR_READ;
				}
			}

			if (futureUse & TextureUse::TEXTURE_SAMPLE) {
				dstStageMask1 |= PipelineStage::FRAGMENT_SHADER;
				dstAccessMask1 |= AccessBit::SHADER_READ;
			}
		}
 	}

	// If image samplers are used during this render pass,
	// add respective barriers before reading them depending on type
	if (this->descriptorType.has_value()) {
		if (this->descriptorType.value() & ImageType::COLOR) {
			srcStageMask0 |= PipelineStage::COLOR_OUTPUT;
			srcAccessMask0 |= AccessBit::COLOR_WRITE;
		if (this->descriptorType.value() & ImageType::DEPTH) {
			srcStageMask0 |= PipelineStage::LATE_FRAGMENT;
			srcAccessMask0 |= AccessBit::DEPTH_STENCIL_WRITE;
		}
		dstStageMask0 |= PipelineStage::FRAGMENT_SHADER;
		dstAccessMask0 |= AccessBit::SHADER_READ;
	}

	std::array<VkSubpassDependency, 2> dependencies{};
	dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
	dependencies[0].srcStageMask = srcStageMask0;
	dependencies[0].srcAccessMask = srcAccessMask0;
	dependencies[0].dstSubpass = 0;
	dependencies[0].dstStageMask = dstStageMask0;
	dependencies[0].dstAccessMask = dstAccessMask0;

	dependencies[1].srcSubpass = 0;
	dependencies[1].srcStageMask = srcStageMask1;
	dependencies[1].srcAccessMask = srcAccessMask1;
	dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
	dependencies[1].dstStageMask = dstStageMask1;
	dependencies[1].dstAccessMask = dstAccessMask1;

	return RenderPass(attachments, references, subpass, dependencies);
}