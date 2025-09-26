#include "PipelineLayout.hpp"

PipelineLayout::PipelineLayout(VulkanWindow* window) : window(window) {}

void PipelineLayout::recreate() {}

VkPipelineLayout PipelineLayout::getHandle() {
	return this->pipelineLayout.handle;
}