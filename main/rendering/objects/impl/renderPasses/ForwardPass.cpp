#include "ForwardPass.hpp"

#include "Error.hpp"
#include "toString.hpp"
#include "../../../../vulkan/VulkanWindow.hpp"
#include "../../../../vulkan/VulkanDevice.hpp"

ForwardPass::ForwardPass(VulkanWindow* window) : RenderPass(window) {
	this->recreate();
}

void ForwardPass::recreate() {
	VkAttachmentDescription attachments[3]{};
	// Output image
	attachments[0].format = VK_FORMAT_R16G16B16A16_SFLOAT;
	attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
	attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
	attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	attachments[0].initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; // Skybox pass transitions texture from UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL
	attachments[0].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	// Output brightness
	attachments[1].format = VK_FORMAT_R16G16B16A16_SFLOAT;
	attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
	attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	attachments[1].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	// Depth buffer
	attachments[2].format = VK_FORMAT_D32_SFLOAT;
	attachments[2].samples = VK_SAMPLE_COUNT_1_BIT;
	attachments[2].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachments[2].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	attachments[2].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	attachments[2].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkAttachmentReference subpassAttachments[2]{};
	subpassAttachments[0].attachment = 0;
	subpassAttachments[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	subpassAttachments[1].attachment = 1;
	subpassAttachments[1].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkAttachmentReference depthAttachment{};
	depthAttachment.attachment = 2;
	depthAttachment.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkSubpassDescription subpasses[1]{};
	subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpasses[0].colorAttachmentCount = 2;
	subpasses[0].pColorAttachments = subpassAttachments;
	subpasses[0].pDepthStencilAttachment = &depthAttachment;

	VkSubpassDependency deps[2]{};
	deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
	deps[0].dstSubpass = 0;
	deps[0].srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	deps[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	deps[0].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	deps[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	deps[0].dependencyFlags = 0;

	deps[1].srcSubpass = 0;
	deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
	deps[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	deps[1].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	deps[1].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	deps[1].dependencyFlags = 0;

	VkRenderPassCreateInfo passInfo{};
	passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	passInfo.attachmentCount = 3;
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

	// Since clear values are determined by the attachments of the render pass
	// we create the clear values here
	VkClearValue colourClearValue{};
	// Clear values are set to (0, 0, 0, 1) since on AMD hardware it is considered
	// a 'fast clear' which are meant to be ~100x faster than regular clears
	// https://gpuopen.com/learn/rdna-performance-guide/
	colourClearValue.color = { {0.0f, 0.0f, 0.0f, 1.0f} };
	VkClearValue depthClearValue{};
	depthClearValue.depthStencil.depth = 1.0f;

	this->clearValues.clear();
	this->clearValues.emplace_back(colourClearValue);
	this->clearValues.emplace_back(colourClearValue);
	this->clearValues.emplace_back(depthClearValue);
}
