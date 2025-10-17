#include "SunPipelineLayout.hpp"

#include "../../../PipelineCreation.hpp"
#include "../../../Uniforms.hpp"

SunPipelineLayout::SunPipelineLayout(VulkanWindow* window, std::map<std::string, vk::DescriptorSetLayout>* descriptorLayouts) 
	: descriptorLayouts(descriptorLayouts), PipelineLayout(window) 
{
	this->recreate();
}

void SunPipelineLayout::recreate() {
	std::vector<VkDescriptorSetLayout> layout;
	layout.emplace_back(this->descriptorLayouts->at("uboVF").handle); // MV matrices
	layout.emplace_back(this->descriptorLayouts->at("uboF").handle); // Inverse view

	std::vector<VkPushConstantRange> pushConstants;

	this->pipelineLayout = createPipelineLayout(*this->window->getDevice(), layout, pushConstants);
}