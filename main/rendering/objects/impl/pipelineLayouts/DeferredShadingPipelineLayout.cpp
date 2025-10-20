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
	layouts.emplace_back(this->descriptorLayouts->at("deferredInputs").handle); // G-buffers
	layouts.emplace_back(this->descriptorLayouts->at("uboVF").handle); // MV matrices
	layouts.emplace_back(this->descriptorLayouts->at("ssboF").handle); // Lights SSBO
	layouts.emplace_back(this->descriptorLayouts->at("uboF").handle); // Inverse view
	layouts.emplace_back(this->descriptorLayouts->at("imageF").handle); // SSAO texture

	std::vector<VkDescriptorSetLayout> shadowLayouts;
	shadowLayouts.emplace_back(this->descriptorLayouts->at("deferredInputs").handle); // G-buffers
	shadowLayouts.emplace_back(this->descriptorLayouts->at("uboVF").handle); // MV matrices
	shadowLayouts.emplace_back(this->descriptorLayouts->at("ssboF").handle); // Lights SSBO
	shadowLayouts.emplace_back(this->descriptorLayouts->at("uboF").handle); // Inverse view
	shadowLayouts.emplace_back(this->descriptorLayouts->at("imageF").handle); // SSAO texture
	shadowLayouts.emplace_back(this->descriptorLayouts->at("imageF").handle); // Point shadow maps
	shadowLayouts.emplace_back(this->descriptorLayouts->at("imageF").handle); // Sun shadow map
	shadowLayouts.emplace_back(this->descriptorLayouts->at("imageF").handle); // Spot shadow maps
	shadowLayouts.emplace_back(this->descriptorLayouts->at("uboF").handle); // Camera planes
	shadowLayouts.emplace_back(this->descriptorLayouts->at("ssboF").handle); // Light matrices SSBO

	VkPushConstantRange lightCount = {
		.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		.size = sizeof(glsl::LightsAndEmissive)
	};

	std::vector<VkPushConstantRange> pushConstants;
	pushConstants.emplace_back(lightCount);

	this->pipelineLayout = createPipelineLayout(*this->window->getDevice(), *this->shadowsEnabled ? shadowLayouts : layouts, pushConstants);
}