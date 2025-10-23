#include "SSAOPipelineLayout.hpp"

#include "../../../PipelineCreation.hpp"

SSAOPipelineLayout::SSAOPipelineLayout(VulkanWindow* window,
	std::map<std::string, vk::DescriptorSetLayout>* descriptorLayouts
) : descriptorLayouts(descriptorLayouts),
	PipelineLayout(window) 
{
	this->recreate();
}

void SSAOPipelineLayout::recreate() {
	std::vector<VkDescriptorSetLayout> layouts;
	layouts.emplace_back(this->descriptorLayouts->at("uboF").handle); // Projective Uniforms
	layouts.emplace_back(this->descriptorLayouts->at("uboF").handle); // SSAO Uniforms
	layouts.emplace_back(this->descriptorLayouts->at("image3F").handle); // Textures

	std::vector<VkPushConstantRange> pushConstants;

	this->pipelineLayout = createPipelineLayout(*this->window->getDevice(), layouts, pushConstants);
}