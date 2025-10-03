#include "LineDebugPipelineLayout.hpp"

#include "../../../PipelineCreation.hpp"

LineDebugPipelineLayout::LineDebugPipelineLayout(
	VulkanWindow* window,
	std::map<std::string, vk::DescriptorSetLayout>* descriptorLayouts) : PipelineLayout(window) {
	this->descriptorLayouts = descriptorLayouts;

	this->recreate();
}

void LineDebugPipelineLayout::recreate() {
	std::vector<VkDescriptorSetLayout> layouts;
	layouts.emplace_back(this->descriptorLayouts->at("uboVF").handle); // MV matrices
	
	std::vector<VkPushConstantRange> pushConstants;

	this->pipelineLayout = createPipelineLayout(*this->window->getDevice(), layouts, pushConstants);
}