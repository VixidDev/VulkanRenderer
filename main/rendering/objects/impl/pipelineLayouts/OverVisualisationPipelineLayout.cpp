#include "OverVisualisationPipelineLayout.hpp"

#include "../../../PipelineCreation.hpp"

OverVisualisationPipelineLayout::OverVisualisationPipelineLayout(VulkanWindow* window,
	std::map<std::string, vk::DescriptorSetLayout>* descriptorLayouts
) : descriptorLayouts(descriptorLayouts),
	PipelineLayout(window) 
{
	this->recreate();
}

void OverVisualisationPipelineLayout::recreate() {
	std::vector<VkDescriptorSetLayout> layouts;
	layouts.emplace_back(this->descriptorLayouts->at("uboVF").handle); // MV matrices

	std::vector<VkPushConstantRange> pushConstants;

	this->pipelineLayout = createPipelineLayout(*this->window->getDevice(), layouts, pushConstants);
}