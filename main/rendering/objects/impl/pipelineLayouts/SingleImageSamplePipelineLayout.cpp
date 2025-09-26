#include "SingleImageSamplePipelineLayout.hpp"

#include "../../../PipelineCreation.hpp"

SingleImageSamplePipelineLayout::SingleImageSamplePipelineLayout(VulkanWindow* window,
	std::map<std::string, vk::DescriptorSetLayout>* descriptorLayouts
) : descriptorLayouts(descriptorLayouts),
	PipelineLayout(window) 
{
	this->recreate();
}

void SingleImageSamplePipelineLayout::recreate() {
	std::vector<VkDescriptorSetLayout> layouts;
	layouts.emplace_back(this->descriptorLayouts->at("imageF").handle); // Input image

	std::vector<VkPushConstantRange> pushConstants;

	this->pipelineLayout = createPipelineLayout(*this->window, layouts, pushConstants);
}