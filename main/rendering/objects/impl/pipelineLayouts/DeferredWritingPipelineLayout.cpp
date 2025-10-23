#include "DeferredWritingPipelineLayout.hpp"

#include "../../../PipelineCreation.hpp"

DeferredWritingPipelineLayout::DeferredWritingPipelineLayout(VulkanWindow* window,
	std::map<std::string, vk::DescriptorSetLayout>* descriptorLayouts
) : descriptorLayouts(descriptorLayouts),
	PipelineLayout(window) 
{
	this->recreate();
}

void DeferredWritingPipelineLayout::recreate() {
	std::vector<VkDescriptorSetLayout> layouts;
	layouts.emplace_back(this->descriptorLayouts->at("uboVF").handle); // MV matrices
	layouts.emplace_back(this->descriptorLayouts->at("image6F").handle); // Materials

	std::vector<VkPushConstantRange> pushConstants;

	this->pipelineLayout = createPipelineLayout(*this->window->getDevice(), layouts, pushConstants);
}