#include "SunViewPipelineLayout.hpp"

#include "../../../PipelineCreation.hpp"
#include "../../../Uniforms.hpp"

SunViewPipelineLayout::SunViewPipelineLayout(VulkanWindow* window,
	std::map<std::string, vk::DescriptorSetLayout>* descriptorLayouts
) : descriptorLayouts(descriptorLayouts),
	PipelineLayout(window) 
{
	this->recreate();
}

void SunViewPipelineLayout::recreate() {
	std::vector<VkDescriptorSetLayout> layout;
	layout.emplace_back(this->descriptorLayouts->at("uboVF").handle); // MV matrices
	layout.emplace_back(this->descriptorLayouts->at("image6F").handle); // Material textures

	std::vector<VkPushConstantRange> pushConstants;

	this->pipelineLayout = createPipelineLayout(*this->window->getDevice(), layout, pushConstants);
}