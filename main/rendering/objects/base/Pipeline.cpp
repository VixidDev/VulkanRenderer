#include "Pipeline.hpp"

Pipeline::Pipeline(VulkanWindow* window) : window(window) {}

void Pipeline::recreate() {}

VkPipeline Pipeline::getHandle() {
	return this->pipeline.handle;
}