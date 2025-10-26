#include "VarianceShadowBlurPass.hpp"

#include "Error.hpp"
#include "toString.hpp"
#include "../../../../vulkan/VulkanWindow.hpp"
#include "../../../../vulkan/VulkanDevice.hpp"

VarianceShadowBlurPass::VarianceShadowBlurPass(VulkanWindow* window) : RenderPass(window) {
	this->recreate();
}

void VarianceShadowBlurPass::recreate() {
	VkAttachmentDescription attachments[1]{};
	// Output image
	attachments[0].format = VK_FORMAT_R32G32_SFLOAT;
	attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
	attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	attachments[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

	VkAttachmentReference swapchainAttachment{};
	swapchainAttachment.attachment = 0;
	swapchainAttachment.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkSubpassDescription subpasses[1]{};
	subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpasses[0].colorAttachmentCount = 1;
	subpasses[0].pColorAttachments = &swapchainAttachment;

	VkSubpassDependency deps[2]{};
	deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
	deps[0].dstSubpass = 0;
	deps[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	deps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	deps[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
	deps[0].dependencyFlags = 0;

	deps[1].srcSubpass = 0;
	deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
	deps[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	deps[1].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	deps[1].dstAccessMask = VK_ACCESS_NONE_KHR;
	deps[1].dependencyFlags = 0;

	VkRenderPassCreateInfo passInfo{};
	passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	passInfo.attachmentCount = 1;
	passInfo.pAttachments = attachments;
	passInfo.subpassCount = 1;
	passInfo.pSubpasses = subpasses;
	passInfo.dependencyCount = 2;
	passInfo.pDependencies = deps;

	VkRenderPass rpass = VK_NULL_HANDLE;
	if (const auto res = vkCreateRenderPass(this->window->getDevice()->getDevice(), &passInfo, nullptr, &rpass); VK_SUCCESS != res) {
		throw Utils::Error("Unable to create render pass\n vkCreateRenderPass() returned %s\n", Utils::toString(res).c_str());
	}

	this->renderPass = vk::RenderPass(this->window->getDevice()->getDevice(), rpass);

	VkClearValue colourClearValue{};
	colourClearValue.color = { {0.0f, 0.0f, 0.0f, 1.0f} };

	this->clearValues.clear();
	this->clearValues.emplace_back(colourClearValue);
}