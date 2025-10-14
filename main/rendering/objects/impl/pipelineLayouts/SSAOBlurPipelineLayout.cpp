#include "SSAOBlurPipelineLayout.hpp"

#include "../../../PipelineCreation.hpp"

SSAOBlurPipelineLayout::SSAOBlurPipelineLayout(VulkanWindow* window,
	std::map<std::string, vk::DescriptorSetLayout>* descriptorLayouts
) : descriptorLayouts(descriptorLayouts),
	PipelineLayout(window) 
{
	this->recreate();
}

void SSAOBlurPipelineLayout::recreate() {
	std::vector<VkDescriptorSetLayout> layouts;
	layouts.emplace_back(this->descriptorLayouts->at("ssaoTextures").handle); // SSAO textures (depth, normals, ssao)
	layouts.emplace_back(this->descriptorLayouts->at("uboF").handle); // Camera planes

	VkPushConstantRange pcr = {
		.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		.size = (sizeof(int) * 2) + (sizeof(float) * 2)
	};

	std::vector<VkPushConstantRange> pushConstants;
	pushConstants.emplace_back(pcr);

	this->pipelineLayout = createPipelineLayout(*this->window->getDevice(), layouts, pushConstants);
}