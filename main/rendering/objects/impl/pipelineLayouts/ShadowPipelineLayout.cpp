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
#if !defined(NDEBUG)
	shadowLayouts.emplace_back(this->descriptorLayouts->at("uboF").handle); // Camera planes
#endif

	VkPushConstantRange depthProjectionMatrix = {
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
		.size = sizeof(glm::mat4)
	};

	std::vector<VkPushConstantRange> pushConstants;
	pushConstants.emplace_back(depthProjectionMatrix);

#if !defined(NDEBUG)
	VkPushConstantRange projectionType = {
	.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
	.offset = depthProjectionMatrix.size,
	.size = sizeof(int)
	};

	pushConstants.emplace_back(projectionType);
#endif

	this->pipelineLayout = createPipelineLayout(*this->window->getDevice(), shadowLayouts, pushConstants);
}