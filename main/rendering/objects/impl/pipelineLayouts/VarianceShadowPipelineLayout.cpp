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
	shadowLayouts.emplace_back(this->descriptorLayouts->at("imageF").handle); // Alpha mask

	VkPushConstantRange lightMatrices = {
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
		.size = sizeof(glm::mat4) * 2
	};

	VkPushConstantRange planeAndLightPos = {
		.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		.offset = lightMatrices.size,
		.size = sizeof(glm::vec4) + sizeof(float)
	};

	std::vector<VkPushConstantRange> pushConstants;
	pushConstants.emplace_back(lightMatrices);
	pushConstants.emplace_back(planeAndLightPos);

	this->pipelineLayout = createPipelineLayout(*this->window->getDevice(), shadowLayouts, pushConstants);
}