#include "DeferredWritingPass.hpp"

#include "Error.hpp"
#include "toString.hpp"
#include "../../../../vulkan/VulkanWindow.hpp"
#include "../../../../vulkan/VulkanDevice.hpp"

DeferredWritingPass::DeferredWritingPass(VulkanWindow* window) : RenderPass(window) {
	this->recreate();
}

void DeferredWritingPass::recreate() {
	VkAttachmentDescription attachments[4]{};
	// G-Buffers
	// normals = rgb
	attachments[0].format = VK_FORMAT_A2R10G10B10_UNORM_PACK32;
	attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
	attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	attachments[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

	// albedo = rgb, roughness = a
	attachments[1].format = VK_FORMAT_R8G8B8A8_UNORM;
	attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
	attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	attachments[1].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

	// emissive = rgb, metalness = a
	attachments[2].format = VK_FORMAT_R8G8B8A8_UNORM;
	attachments[2].samples = VK_SAMPLE_COUNT_1_BIT;
	attachments[2].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachments[2].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	attachments[2].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	attachments[2].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

	// Depth buffer
	attachments[3].format = VK_FORMAT_D32_SFLOAT;
	attachments[3].samples = VK_SAMPLE_COUNT_1_BIT;
	attachments[3].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachments[3].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	attachments[3].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	attachments[3].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

	VkAttachmentReference gBufferAttachments[3]{};
	gBufferAttachments[0].attachment = 0;
	gBufferAttachments[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	gBufferAttachments[1].attachment = 1;
	gBufferAttachments[1].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	gBufferAttachments[2].attachment = 2;
	gBufferAttachments[2].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkAttachmentReference depthAttachment{};
	depthAttachment.attachment = 3;
	depthAttachment.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkSubpassDescription subpasses[1]{};
	subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpasses[0].colorAttachmentCount = 3;
	subpasses[0].pColorAttachments = gBufferAttachments;
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
	deps[1].srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	deps[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	deps[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	deps[1].dependencyFlags = 0;

	VkRenderPassCreateInfo passInfo{};
	passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	passInfo.attachmentCount = 4;
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
	this->clearValues.emplace_back(colourClearValue);
	this->clearValues.emplace_back(depthClearValue);
}