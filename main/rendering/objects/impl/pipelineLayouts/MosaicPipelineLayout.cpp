#include "MosaicPipelineLayout.hpp"

#include "../../../PipelineCreation.hpp"

MosaicPipelineLayout::MosaicPipelineLayout(VulkanWindow* window,
	std::map<std::string, vk::DescriptorSetLayout>* descriptorLayouts
) : descriptorLayouts(descriptorLayouts),
	PipelineLayout(window) {
	this->recreate();
}

void MosaicPipelineLayout::recreate() {
	std::vector<VkDescriptorSetLayout> layouts;
	layouts.emplace_back(this->descriptorLayouts->at("imageF").handle); // Input image

	std::vector<VkPushConstantRange> pushConstants;

	this->pipelineLayout = createPipelineLayout(*this->window->getDevice(), layouts, pushConstants);
}