#include "CubemapShadowPipelineLayout.hpp"

#include "../../../PipelineCreation.hpp"

#include <glm/mat4x4.hpp>

CubemapShadowPipelineLayout::CubemapShadowPipelineLayout(
	VulkanWindow* window,
	std::map<std::string, vk::DescriptorSetLayout>* descriptorLayouts) : PipelineLayout(window) {
	this->descriptorLayouts = descriptorLayouts;

	this->recreate();
}

void CubemapShadowPipelineLayout::recreate() {
	std::vector<VkDescriptorSetLayout> shadowLayouts;

	VkPushConstantRange depthProjectionMatrix = {
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
		.size = sizeof(glm::mat4)
	};

	VkPushConstantRange planeAndLightPos = {
		.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		.offset = sizeof(glm::mat4),
		// The push constant in the shader is actually a vec3 and a float, but since vec3's have
		// a member alignment of 16 they will actually take up 16 bytes instead of 12. So the push constant
		// actually needs 16 (vec4) + 4 (float) = 20 bytes of size in order to create the pipeline, otherwise
		// the validation layers will complain, rightly so. If we were tight on memory budget for the push constant
		// we could pack the float into the free 4 bytes reserved by the vec3, but since we are not ill keep them 
		// seperate for now.
		.size = sizeof(glm::vec4) + sizeof(float)
	};

	std::vector<VkPushConstantRange> pushConstants;
	pushConstants.emplace_back(depthProjectionMatrix);
	pushConstants.emplace_back(planeAndLightPos);

	this->pipelineLayout = createPipelineLayout(*this->window, shadowLayouts, pushConstants);
}