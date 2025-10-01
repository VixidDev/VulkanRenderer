#include "DebugViewsPipelineLayout.hpp"

#include "../../../PipelineCreation.hpp"

DebugViewsPipelineLayout::DebugViewsPipelineLayout(VulkanWindow* window,
	std::map<std::string, vk::DescriptorSetLayout>* descriptorLayouts
) : descriptorLayouts(descriptorLayouts),
	PipelineLayout(window) 
{
	this->recreate();
}

void DebugViewsPipelineLayout::recreate() {
	std::vector<VkDescriptorSetLayout> layouts;
	layouts.emplace_back(this->descriptorLayouts->at("uboVF").handle);     // MV matrices
	layouts.emplace_back(this->descriptorLayouts->at("ssboF").handle);	   // Lights SSBO
	layouts.emplace_back(this->descriptorLayouts->at("uboF").handle);	   // Camera planes
	layouts.emplace_back(this->descriptorLayouts->at("materials").handle); // Material textures

	VkPushConstantRange debugState = {
		.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		.size = sizeof(debugStatePC)
	};

	std::vector<VkPushConstantRange> pushConstants;
	pushConstants.emplace_back(debugState);

	this->pipelineLayout = createPipelineLayout(*this->window, layouts, pushConstants);
}