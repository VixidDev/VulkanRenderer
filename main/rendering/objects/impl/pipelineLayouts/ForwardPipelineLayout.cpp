#include "ForwardPipelineLayout.hpp"

#include "../../../PipelineCreation.hpp"
#include "../../../Uniforms.hpp"

ForwardPipelineLayout::ForwardPipelineLayout(VulkanWindow* window,
	std::map<std::string, vk::DescriptorSetLayout>* descriptorLayouts,
	bool* shadowsEnabled
) : descriptorLayouts(descriptorLayouts),
	shadowsEnabled(shadowsEnabled), 
	PipelineLayout(window)
{
	this->recreate();
}

void ForwardPipelineLayout::recreate() {
	std::vector<VkDescriptorSetLayout> forwardLayouts;
	forwardLayouts.emplace_back(this->descriptorLayouts->at("uboVF").handle); // MV matrices
	forwardLayouts.emplace_back(this->descriptorLayouts->at("ssboF").handle); // Lights SSBO
	forwardLayouts.emplace_back(this->descriptorLayouts->at("imageF").handle); // SSAO texture
	forwardLayouts.emplace_back(this->descriptorLayouts->at("materials").handle); // Material textures

	std::vector<VkDescriptorSetLayout> forwardShadowLayouts;
	forwardShadowLayouts.emplace_back(this->descriptorLayouts->at("uboVF").handle); // MV matrices
	forwardShadowLayouts.emplace_back(this->descriptorLayouts->at("ssboF").handle); // Lights SSBO
	forwardShadowLayouts.emplace_back(this->descriptorLayouts->at("imageF").handle); // SSAO texture
	forwardShadowLayouts.emplace_back(this->descriptorLayouts->at("imageF").handle); // Point shadow maps
	forwardShadowLayouts.emplace_back(this->descriptorLayouts->at("imageF").handle); // Directional shadow maps
	forwardShadowLayouts.emplace_back(this->descriptorLayouts->at("imageF").handle); // Spot shadow maps
	forwardShadowLayouts.emplace_back(this->descriptorLayouts->at("uboF").handle); // Camera planes
	forwardShadowLayouts.emplace_back(this->descriptorLayouts->at("ssboF").handle); // Light matrices SSBO
	forwardShadowLayouts.emplace_back(this->descriptorLayouts->at("materials").handle); // Material textures

	VkPushConstantRange lightCount = {
		.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		.size = sizeof(glsl::LightsAndEmissive)
	};

	std::vector<VkPushConstantRange> pushConstants;
	pushConstants.emplace_back(lightCount);

	this->pipelineLayout = createPipelineLayout(*this->window->getDevice(), *this->shadowsEnabled ? forwardShadowLayouts : forwardLayouts, pushConstants);
}