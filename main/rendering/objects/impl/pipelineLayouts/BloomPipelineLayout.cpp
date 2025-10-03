#include "BloomPipelineLayout.hpp"

#include "../../../PipelineCreation.hpp"

BloomPipelineLayout::BloomPipelineLayout(VulkanWindow* window,
	std::map<std::string, vk::DescriptorSetLayout>* descriptorLayouts
) : descriptorLayouts(descriptorLayouts),
	PipelineLayout(window) 
{
	this->recreate();
}

void BloomPipelineLayout::recreate() {
	std::vector<VkDescriptorSetLayout> layouts;
	layouts.emplace_back(this->descriptorLayouts->at("imageF").handle); // Input image

	VkPushConstantRange directionPushConstant = {
		.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		.size = sizeof(int)
	};

	std::vector<VkPushConstantRange> pushConstants;
	pushConstants.emplace_back(directionPushConstant);

	this->pipelineLayout = createPipelineLayout(*this->window->getDevice(), layouts, pushConstants);
}