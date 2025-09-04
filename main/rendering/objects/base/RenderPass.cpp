#include "RenderPass.hpp"

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