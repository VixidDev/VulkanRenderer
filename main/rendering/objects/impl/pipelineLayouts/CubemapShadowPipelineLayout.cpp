#include "CubemapShadowPipelineLayout.hpp"

#include "../../../PipelineCreation.hpp"

CubemapShadowPipelineLayout::CubemapShadowPipelineLayout(
	VulkanWindow* window,
	std::map<std::string, vk::DescriptorSetLayout>* descriptorLayouts) : PipelineLayout(window) {
	this->descriptorLayouts = descriptorLayouts;

	this->recreate();
}

void CubemapShadowPipelineLayout::recreate() {
	std::vector<VkDescriptorSetLayout> shadowLayouts;
	shadowLayouts.emplace_back(this->descriptorLayouts->at("uboV").handle); // Depth matrix
	shadowLayouts.emplace_back(this->descriptorLayouts->at("uboF").handle); // Camera planes

	std::vector<VkPushConstantRange> emptyPushConstants;

	this->pipelineLayout = createPipelineLayout(*this->window, shadowLayouts, emptyPushConstants);
}