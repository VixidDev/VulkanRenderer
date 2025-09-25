#include "DeferredShadingPipelineLayout.hpp"

#include "../../../PipelineCreation.hpp"
#include "../../../Uniforms.hpp"

DeferredShadingPipelineLayout::DeferredShadingPipelineLayout(VulkanWindow* window,
	std::map<std::string, vk::DescriptorSetLayout>* descriptorLayouts,
	bool* shadowsEnabled
) : descriptorLayouts(descriptorLayouts),
	shadowsEnabled(shadowsEnabled),
	PipelineLayout(window) 
{
	this->recreate();
}

void DeferredShadingPipelineLayout::recreate() {
	std::vector<VkDescriptorSetLayout> layouts;
	layouts.emplace_back(this->descriptorLayouts->at("deferredInputAttachments").handle); // Input attachments
	layouts.emplace_back(this->descriptorLayouts->at("uboVF").handle); // MV matrices
	layouts.emplace_back(this->descriptorLayouts->at("ssboF").handle); // Lights SSBO

	std::vector<VkDescriptorSetLayout> shadowLayouts;
	shadowLayouts.emplace_back(this->descriptorLayouts->at("deferredInputAttachments").handle); // Input attachments
	shadowLayouts.emplace_back(this->descriptorLayouts->at("uboVF").handle); // MV matrices
	shadowLayouts.emplace_back(this->descriptorLayouts->at("imageF").handle); // Point shadow maps
	shadowLayouts.emplace_back(this->descriptorLayouts->at("imageF").handle); // Sun shadow map
	shadowLayouts.emplace_back(this->descriptorLayouts->at("uboF").handle); // Camera planes
	shadowLayouts.emplace_back(this->descriptorLayouts->at("ssboF").handle); // Lights SSBO
	shadowLayouts.emplace_back(this->descriptorLayouts->at("ssboF").handle); // Light matrices SSBO

	VkPushConstantRange lightCount = {
		.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		.size = sizeof(glsl::LightsAndEmissive)
	};

	std::vector<VkPushConstantRange> pushConstants;
	pushConstants.emplace_back(lightCount);

	this->pipelineLayout = createPipelineLayout(*this->window, *this->shadowsEnabled ? shadowLayouts : layouts, pushConstants);
}