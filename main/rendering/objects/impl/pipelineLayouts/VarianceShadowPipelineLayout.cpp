#include "VarianceShadowPipelineLayout.hpp"

#include "../../../PipelineCreation.hpp"

#include <glm/mat4x4.hpp>

VarianceShadowPipelineLayout::VarianceShadowPipelineLayout(
	VulkanWindow* window,
	std::map<std::string, vk::DescriptorSetLayout>* descriptorLayouts) : PipelineLayout(window) {
	this->descriptorLayouts = descriptorLayouts;

	this->recreate();
}

void VarianceShadowPipelineLayout::recreate() {
	std::vector<VkDescriptorSetLayout> shadowLayouts;
	shadowLayouts.emplace_back(this->descriptorLayouts->at("uboF").handle);   // Camera planes
	shadowLayouts.emplace_back(this->descriptorLayouts->at("imageF").handle); // Alpha mask

	VkPushConstantRange lightMatrices = {
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
		.size = sizeof(glm::mat4) * 2
	};

	std::vector<VkPushConstantRange> pushConstants;
	pushConstants.emplace_back(lightMatrices);

	this->pipelineLayout = createPipelineLayout(*this->window->getDevice(), shadowLayouts, pushConstants);
}