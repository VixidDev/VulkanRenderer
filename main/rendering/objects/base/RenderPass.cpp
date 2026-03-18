#include "RenderPass.hpp"

#include "error.hpp"
#include "../../../vulkan/VkUtils.hpp"

RenderPass::RenderPass(VulkanWindow* window) : window(window) {}

void RenderPass::recreate() {}

vk::RenderPass& RenderPass::getRenderPass() {
	return this->renderPass;
}

VkRenderPass RenderPass::getRenderPassHandle() {
	return this->renderPass.handle;
}

std::vector<VkClearValue>& RenderPass::getClearValues() {
	return this->clearValues;
}

RenderPass::Builder* RenderPass::Builder::withColourAttachment(
	Texture colourAttachment, AttachmentLoadOp loadOp, AttachmentStoreOp storeOp, ImageLayout layout
) {
	AttachmentDesc desc = { colourAttachment, loadOp, storeOp, layout };

	this->attachments.emplace_back(desc);
	return this;
}

RenderPass::Builder* RenderPass::Builder::withDepthAttachment(
	Texture depthAttachment, AttachmentLoadOp loadOp, AttachmentStoreOp storeOp, ImageLayout layout
) {
	if (this->depthTextureIndex.has_value())
		throw Utils::Error("Render passes can only have 1 depth buffer bound!\n");
	
	AttachmentDesc desc = { depthAttachment, loadOp, storeOp, layout };

	this->attachments.emplace_back(desc);
	this->depthTextureIndex = this->attachments.size() - 1;
	return this;
}

RenderPass::Builder* RenderPass::Builder::usesDescriptorInShader(ImageType imageType) {
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
		attachmentDesc.initialLayout = (VkImageLayout)desc.initialLayout.value_or(VK_IMAGE_LAYOUT_UNDEFINED);
		attachmentDesc.finalLayout = (VkImageLayout)desc.finalLayout.value_or(VkUtils::getFinalLayoutFromFormat(texture.getFormat()));

		attachments.emplace_back(attachmentDesc);
	}

	// Attachment references
	std::vector<VkAttachmentReference> references;
	references.reserve(this->attachments.size());

	for (size_t i = 0; i < this->attachments.size(); i++) {
		AttachmentDesc& desc = this->attachments.at(i);
		TextureBuffer& texture = Textures::get(desc.texture);

		VkAttachmentReference reference{};
		reference.attachment = i;
		reference.layout = (VkImageLayout)desc.finalLayout.value_or(VkUtils::getFinalLayoutFromFormat(texture.getFormat()));

		references.emplace_back(reference);
	}

	VkSubpassDescription subpass{};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = this->attachments.size() - (this->depthTextureIndex ? 1 : 0);
	subpass.pColorAttachments = references.data();
	subpass.pDepthStencilAttachment = this->depthTextureIndex ? &references.at(this->depthTextureIndex.value()) : nullptr;

	// Determine stage and access masks

	// If an attachment has a clear load op, that is considered a write operation and needs to be barrier'd
	// i.e. if a depth attachment is written to in a previous pass, then cleared and written to in the 
	// current subpass, the srcStageMask needs LATE_FRAGMENT_TESTS since store ops occur there for depth attachments,
	// then dstStageMask needs EARLY_FRAGMENT_TESTS since load ops occur there

	PipelineStageFlags srcStageMask0 = PipelineStage::NONE;
	PipelineStageFlags dstStageMask0 = PipelineStage::NONE;
	AccessFlags srcAccessMask0 = AccessBit::NONE;
	AccessFlags dstAccessMask0 = AccessBit::NONE;

	for (size_t i = 0; i < this->attachments.size(); i++) {
		AttachmentDesc& desc = this->attachments.at(i);
		TextureBuffer& texture = Textures::get(desc.texture);

		// If we have attachments with a loadOp of CLEAR, we want to synchronise
		// on depth/color write, since CLEARs are considered write ops
		if (desc.loadOp == AttachmentLoadOp::CLEAR) {
			if (Textures::isOfDepthFormat(texture.getFormat())) {
				dstStageMask0 |= PipelineStage::EARLY_FRAGMENT;
				dstAccessMask0 |= AccessBit::DEPTH_STENCIL_WRITE;
			} else {
				dstStageMask0 |= PipelineStage::COLOR_OUTPUT;
				dstAccessMask0 |= AccessBit::COLOR_WRITE;
			}
		}

		// If we have attachments with a loadOp of LOAD, we want to synchronise 
		// on depth/color load for that attachment type
		if (desc.loadOp == AttachmentLoadOp::LOAD) {
			if (Textures::isOfDepthFormat(texture.getFormat())) {
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
			if (desc.finalLayout.has_value() && Textures::isOfDepthLayout(desc.finalLayout.value())) {
				srcStageMask0 |= PipelineStage::EARLY_FRAGMENT | PipelineStage::LATE_FRAGMENT;
				srcAccessMask0 |= AccessBit::DEPTH_STENCIL_WRITE;
			} else if (desc.finalLayout.has_value() && Textures::isOfColorLayout(desc.finalLayout.value())) {
				srcStageMask0 |= PipelineStage::COLOR_OUTPUT;
				srcAccessMask0 |= AccessBit::COLOR_WRITE;
			}
		}

		// We need to synchronise on any work done this render pass
		if (Textures::isOfDepthFormat(texture.getFormat())) {
			dstStageMask0 |= PipelineStage::EARLY_FRAGMENT;
			// It would be more correct to make the DEPTH_STENCIL dstAccessMasks present depending on the 
			// pipeline depth write/test flags, but it would be more compilcated since multiple pipelines 
			// can use the same render pass, and I don't want to make the same pass with just differing access masks
			dstAccessMask0 |= AccessBit::DEPTH_STENCIL_WRITE | AccessBit::DEPTH_STENCIL_READ;
		} else {
			dstStageMask0 |= PipelineStage::COLOR_OUTPUT;
			dstAccessMask0 |= AccessBit::COLOR_WRITE;
		}
 	}

	// If image samplers are used during this render pass,
	// add respective barriers before reading them depending on type
	if (this->descriptorType.has_value()) {
		if (this->descriptorType == ImageType::COLOR) {
			srcStageMask0 |= PipelineStage::COLOR_OUTPUT;
			srcAccessMask0 |= AccessBit::COLOR_WRITE;
		} else if (this->descriptorType == ImageType::DEPTH) {
			srcStageMask0 |= PipelineStage::LATE_FRAGMENT;
			srcAccessMask0 |= AccessBit::DEPTH_STENCIL_WRITE;
		}
		dstStageMask0 |= PipelineStage::FRAGMENT_SHADER;
		dstAccessMask0 |= AccessBit::SHADER_READ;
	}

	VkSubpassDependency dependencies[2]{};
	// The previous subpass...
	dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
	// ...must finish the following stages...
	dependencies[0].srcStageMask = srcStageMask0;
	// ...with the following access...
	dependencies[0].srcAccessMask = srcAccessMask0;
	// ...before allowing this subpass..
	dependencies[0].dstSubpass = 0;
	// ...with the following stages...
	dependencies[0].dstStageMask = dstStageMask0;
	// ...and the following access to take place.
	dependencies[0].dstAccessMask = dstAccessMask0;


}