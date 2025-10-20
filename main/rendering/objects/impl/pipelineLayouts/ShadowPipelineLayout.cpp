#include "ShadowPipelineLayout.hpp"

#include "../../../PipelineCreation.hpp"

#include <glm/mat4x4.hpp>

ShadowPipelineLayout::ShadowPipelineLayout(
	VulkanWindow* window,
	std::map<std::string, vk::DescriptorSetLayout>* descriptorLayouts) : PipelineLayout(window) 
{
	this->descriptorLayouts = descriptorLayouts;

	this->recreate();
}

void ShadowPipelineLayout::recreate() {
	std::vector<VkDescriptorSetLayout> shadowLayouts;
	shadowLayouts.emplace_back(this->descriptorLayouts->at("uboF").handle);   // Camera planes
	shadowLayouts.emplace_back(this->descriptorLayouts->at("imageF").handle); // Alpha mask

	VkPushConstantRange depthProjectionMatrix = {
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
		.size = sizeof(glm::mat4)
	};
	VkPushConstantRange projectionType = {
		.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		.offset = depthProjectionMatrix.size,
		.size = sizeof(int)
	};

	std::vector<VkPushConstantRange> pushConstants;
	pushConstants.emplace_back(depthProjectionMatrix);
	pushConstants.emplace_back(projectionType);

	this->pipelineLayout = createPipelineLayout(*this->window->getDevice(), shadowLayouts, pushConstants);
}