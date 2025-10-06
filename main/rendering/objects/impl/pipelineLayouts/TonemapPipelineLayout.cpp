#include "TonemapPipelineLayout.hpp"

#include "../../../PipelineCreation.hpp"

TonemapPipelineLayout::TonemapPipelineLayout(VulkanWindow* window, std::map<std::string, vk::DescriptorSetLayout>* descriptorLayouts
) : descriptorLayouts(descriptorLayouts),
	PipelineLayout(window) 
{
	this->recreate();
}

void TonemapPipelineLayout::recreate() {
	std::vector<VkDescriptorSetLayout> layouts;
	layouts.emplace_back(this->descriptorLayouts->at("imageF").handle); // Input image

	VkPushConstantRange pcr = {
		.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		.size = sizeof(int) + sizeof(float)
	};

	std::vector<VkPushConstantRange> pushConstants;
	pushConstants.emplace_back(pcr);

	this->pipelineLayout = createPipelineLayout(*this->window->getDevice(), layouts, pushConstants);
}