#include "CompositionPipelineLayout.hpp"

#include "../../../PipelineCreation.hpp"

CompositionPipelineLayout::CompositionPipelineLayout(VulkanWindow* window,
	std::map<std::string, vk::DescriptorSetLayout>* descriptorLayouts
) : descriptorLayouts(descriptorLayouts),
	PipelineLayout(window) {
	this->recreate();
}

void CompositionPipelineLayout::recreate() {
	std::vector<VkDescriptorSetLayout> layouts;
	layouts.emplace_back(this->descriptorLayouts->at("imageF").handle); // Input image
	layouts.emplace_back(this->descriptorLayouts->at("imageF").handle); // Input image

	std::vector<VkPushConstantRange> pushConstants;

	this->pipelineLayout = createPipelineLayout(*this->window, layouts, pushConstants);
}