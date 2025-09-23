#include "DeferredPass.hpp"

#include "Error.hpp"
#include "toString.hpp"
#include "../../../../vulkan/VulkanWindow.hpp"
#include "../../../../vulkan/VulkanDevice.hpp"

DeferredPass::DeferredPass(VulkanWindow* window, VkSampleCountFlagBits* sampleCount) : RenderPass(window) {
	this->sampleCount = sampleCount;

	this->recreate();
}

void DeferredPass::recreate() {
	VkAttachmentDescription attachments[4]{};
	// Swapchain image
	attachments[0].format = this->window->swapchainFormat;
	attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
	attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

	// G-Buffers
	// normals = rgb, metalness = a
	attachments[1].format = VK_FORMAT_R16G16B16A16_SFLOAT;
	attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
	attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	attachments[1].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

	// albedo = rgb, roughness = a
	attachments[2].format = VK_FORMAT_R16G16B16A16_SFLOAT;
	attachments[2].samples = VK_SAMPLE_COUNT_1_BIT;
	attachments[2].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachments[2].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	attachments[2].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	attachments[2].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

	// Depth buffer
	attachments[3].format = VK_FORMAT_D32_SFLOAT;
	attachments[3].samples = VK_SAMPLE_COUNT_1_BIT;
	attachments[3].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachments[3].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	attachments[3].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	attachments[3].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkAttachmentReference swapchainAttachment{};
	swapchainAttachment.attachment = 0;
	swapchainAttachment.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkAttachmentReference gBufferAttachments[2]{};
	gBufferAttachments[0].attachment = 1;
	gBufferAttachments[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	gBufferAttachments[1].attachment = 2;
	gBufferAttachments[1].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkAttachmentReference depthAttachment{};
	depthAttachment.attachment = 3;
	depthAttachment.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkAttachmentReference inputAttachments[3]{};
	inputAttachments[0].attachment = 1;
	inputAttachments[0].layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	inputAttachments[1].attachment = 2;
	inputAttachments[1].layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	inputAttachments[2].attachment = 3;
	inputAttachments[2].layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

	VkSubpassDescription subpasses[2]{};
	subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpasses[0].colorAttachmentCount = 2;
	subpasses[0].pColorAttachments = gBufferAttachments;
	subpasses[0].pDepthStencilAttachment = &depthAttachment;

	subpasses[1].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpasses[1].colorAttachmentCount = 1;
	subpasses[1].pColorAttachments = &swapchainAttachment;
	subpasses[1].inputAttachmentCount = 3;
	subpasses[1].pInputAttachments = inputAttachments;

	VkSubpassDependency deps[3]{};
	deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
	deps[0].dstSubpass = 0;
	deps[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	deps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	deps[0].srcAccessMask = VK_ACCESS_NONE_KHR;
	deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	deps[0].dependencyFlags = 0;

	deps[1].srcSubpass = 0;
	deps[1].dstSubpass = 1;
	deps[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	deps[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	deps[1].dstAccessMask = VK_ACCESS_INPUT_ATTACHMENT_READ_BIT;
	deps[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

	deps[2].srcSubpass = 0;
	deps[2].dstSubpass = VK_SUBPASS_EXTERNAL;
	deps[2].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	deps[2].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	deps[2].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	deps[2].dstAccessMask = VK_ACCESS_NONE_KHR;
	deps[2].dependencyFlags = 0;

	VkRenderPassCreateInfo passInfo{};
	passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	passInfo.attachmentCount = 4;
	passInfo.pAttachments = attachments;
	passInfo.subpassCount = 2;
	passInfo.pSubpasses = subpasses;
	passInfo.dependencyCount = 3;
	passInfo.pDependencies = deps;

	VkRenderPass rpass = VK_NULL_HANDLE;
	if (const auto res = vkCreateRenderPass(this->window->device->device, &passInfo, nullptr, &rpass); VK_SUCCESS != res) {
		throw Utils::Error("Unable to create render pass\n vkCreateRenderPass() returned %s\n", Utils::toString(res).c_str());
	}

	this->renderPass = vk::RenderPass(this->window->device->device, rpass);

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