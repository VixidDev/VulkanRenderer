#include "ForwardPipelineLayout.hpp"

#include "../../../PipelineCreation.hpp"

ForwardPipelineLayout::ForwardPipelineLayout(
	VulkanWindow* window,
	std::map<std::string, vk::DescriptorSetLayout>* descriptorLayouts,
	bool* shadowsEnabled) : PipelineLayout(window) 
{
	this->descriptorLayouts = descriptorLayouts;
	this->shadowsEnabled = shadowsEnabled;

	this->recreate();
}

void ForwardPipelineLayout::recreate() {
	std::vector<VkDescriptorSetLayout> forwardLayouts;
	forwardLayouts.emplace_back(this->descriptorLayouts->at("uboVF").handle); // MV matrices
	forwardLayouts.emplace_back(this->descriptorLayouts->at("materials").handle); // Material textures

	std::vector<VkDescriptorSetLayout> forwardShadowLayouts;
	forwardShadowLayouts.emplace_back(this->descriptorLayouts->at("uboVF").handle); // MV matrices
	forwardShadowLayouts.emplace_back(this->descriptorLayouts->at("materials").handle); // Material textures
	forwardShadowLayouts.emplace_back(this->descriptorLayouts->at("uboV").handle); // Depth matrix
	forwardShadowLayouts.emplace_back(this->descriptorLayouts->at("imageF").handle); // Shadow map

	std::vector<VkPushConstantRange> emptyPushConstant;

	this->pipelineLayout = createPipelineLayout(*this->window, *this->shadowsEnabled ? forwardShadowLayouts : forwardLayouts, emptyPushConstant);
}