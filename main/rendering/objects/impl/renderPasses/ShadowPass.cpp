#include "ShadowPass.hpp"

#include "Error.hpp"
#include "toString.hpp"
#include "../../../../vulkan/VulkanWindow.hpp"
#include "../../../../vulkan/VulkanDevice.hpp"

ShadowPass::ShadowPass(VulkanWindow* window, VkSampleCountFlagBits* sampleCount) : RenderPass(window) {
	this->sampleCount = sampleCount;

	this->recreate();
}

void ShadowPass::recreate() {
#if !defined(NDEBUG)
	const int ATTACHMENTS = 2;
#else
	const int ATTACHMENTS = 1;
#endif
	VkAttachmentDescription attachments[ATTACHMENTS]{};
	attachments[0].format = VK_FORMAT_D32_SFLOAT;
	attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
	attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	attachments[0].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

#if !defined(NDEBUG)
	// Attachment for writing linear depth to so we can visualise it with ImGUI
	attachments[1].format = VK_FORMAT_R16G16B16A16_SFLOAT;
	attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
	attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	attachments[1].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
#endif

	VkAttachmentReference depthAttachment{};
	depthAttachment.attachment = 0;
	depthAttachment.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

#if !defined(NDEBUG)
	VkAttachmentReference linearDepthAttachment{};
	linearDepthAttachment.attachment = 1;
	linearDepthAttachment.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
#endif

	VkSubpassDescription subpasses[1]{};
	subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
#if !defined(NDEBUG)
	subpasses[0].colorAttachmentCount = 1;
	subpasses[0].pColorAttachments = &linearDepthAttachment;
#else
	subpasses[0].colorAttachmentCount = 0;
#endif
	subpasses[0].pDepthStencilAttachment = &depthAttachment;

	VkSubpassDependency deps[2]{};
	deps[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
	deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
	deps[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
	deps[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	deps[0].dstSubpass = 0;
	deps[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	deps[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;

	deps[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
	deps[1].srcSubpass = 0;
	deps[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	deps[1].srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
	deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
	deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	deps[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

	VkRenderPassCreateInfo passInfo{};
	passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	passInfo.attachmentCount = ATTACHMENTS;
	passInfo.pAttachments = attachments;
	passInfo.subpassCount = 1;
	passInfo.pSubpasses = subpasses;
	passInfo.dependencyCount = 2;
	passInfo.pDependencies = deps;

	VkRenderPass rpass = VK_NULL_HANDLE;
	if (const auto res = vkCreateRenderPass(this->window->device->device, &passInfo, nullptr, &rpass); VK_SUCCESS != res) {
		throw Utils::Error("Unable to create render pass\n vkCreateRenderPass() returned %s\n", Utils::toString(res).c_str());
	}

	this->renderPass = vk::RenderPass(this->window->device->device, rpass);

	VkClearValue depthClearValue{};
	depthClearValue.depthStencil.depth = 1.0f;
#if !defined(NDEBUG)
	VkClearValue colourClearValue{};
	colourClearValue.color = { {0.0f, 0.0f, 0.0f, 1.0f} };
#endif

	this->clearValues.clear();
	this->clearValues.emplace_back(depthClearValue);
#if !defined(NDEBUG)
	this->clearValues.emplace_back(colourClearValue);
#endif
}