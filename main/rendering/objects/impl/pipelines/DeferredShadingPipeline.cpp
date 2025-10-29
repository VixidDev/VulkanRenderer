#include "DeferredShadingPipeline.hpp"

#include "Error.hpp"
#include "toString.hpp"
#include "../../../../vulkan/VulkanDevice.hpp"
#include "../../../../vulkan/Swapchain.hpp"
#include "../../../PipelineCreation.hpp"

DeferredShadingPipeline::DeferredShadingPipeline(
	VulkanWindow* window,
	PipelineLayout* pipelineLayout,
	RenderPass* renderPass,
	bool* shadowsEnabled,
	int* vsmShadowsEnabled,
	bool* useViewSpaceNormals,
	int* numLights
) : shadowsEnabled(shadowsEnabled),
	Pipeline(window) 
{
	this->pipelineLayout = pipelineLayout;
	this->renderPass = renderPass;
	this->vsmShadowsEnabled = vsmShadowsEnabled;
	this->useViewSpaceNormals = useViewSpaceNormals;
	this->numLights = numLights;

	this->renderExtent = &this->window->getSwapchain()->getExtent();

	this->recreate();
}

void DeferredShadingPipeline::recreate() {
	vk::ShaderModule vert = loadShaderModule(*this->window->getDevice(), "assets/main/shaders/fullScreen.vert.spv");
	vk::ShaderModule frag;

	if (*this->shadowsEnabled) {
		if (*this->vsmShadowsEnabled) {
			frag = loadShaderModule(*this->window->getDevice(), "assets/main/shaders/deferredShadowVSM.frag.spv");
		} else {
			frag = loadShaderModule(*this->window->getDevice(), "assets/main/shaders/deferredShadow.frag.spv");
		}
	} else {
		frag = loadShaderModule(*this->window->getDevice(), "assets/main/shaders/deferred.frag.spv");
	}

	this->viewSpaceNormals = *this->useViewSpaceNormals ? 1 : 0;

	struct SpecializationData {
		int viewSpaceNormals{};
		int numLights{};
	};

	SpecializationData specializationData = {
		.viewSpaceNormals = this->viewSpaceNormals,
		.numLights = *this->numLights
	};

	// layout(constant_id = 0) const int VIEW_SPACE_NORMALS = 0;
	// layout(constant_id = 1) const int NUM_LIGHTS = 0;
	VkSpecializationMapEntry specializationMapEntries[2]{};
	specializationMapEntries[0].constantID = 0;
	specializationMapEntries[0].offset = 0;
	specializationMapEntries[0].size = sizeof(int);
	specializationMapEntries[1].constantID = 1;
	specializationMapEntries[1].offset = sizeof(int);
	specializationMapEntries[1].size = sizeof(int);

	VkSpecializationInfo specializationInfo = {
		.mapEntryCount = 2,
		.pMapEntries = specializationMapEntries,
		.dataSize = sizeof(SpecializationData),
		.pData = &specializationData
	};

	VkPipelineShaderStageCreateInfo stages[2]{};
	stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
	stages[0].module = vert.handle;
	stages[0].pName = "main";

	stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	stages[1].module = frag.handle;
	stages[1].pName = "main";
	stages[1].pSpecializationInfo = &specializationInfo;

	VkPipelineVertexInputStateCreateInfo inputInfo{};
	inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

	VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
	assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	assemblyInfo.primitiveRestartEnable = VK_FALSE;

	VkViewport viewport{};
	viewport.x = 0.0f;
	viewport.y = 0.0f;
	viewport.width = static_cast<float>(this->renderExtent->width);
	viewport.height = static_cast<float>(this->renderExtent->height);
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;

	VkRect2D scissor{};
	scissor.offset = VkOffset2D{ 0, 0 };
	scissor.extent = *this->renderExtent;

	VkPipelineViewportStateCreateInfo viewportInfo{};
	viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportInfo.viewportCount = 1;
	viewportInfo.pViewports = &viewport;
	viewportInfo.scissorCount = 1;
	viewportInfo.pScissors = &scissor;

	VkPipelineRasterizationStateCreateInfo rasterInfo{};
	rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterInfo.depthBiasEnable = VK_FALSE;
	rasterInfo.rasterizerDiscardEnable = VK_FALSE;
	rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
	rasterInfo.cullMode = VK_CULL_MODE_NONE;
	rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	rasterInfo.depthBiasEnable = VK_FALSE;
	rasterInfo.lineWidth = 1.0f;

	VkPipelineMultisampleStateCreateInfo multisampleInfo{};
	multisampleInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisampleInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

	VkPipelineColorBlendAttachmentState blendStates[2]{};
	blendStates[0].blendEnable = VK_FALSE;
	blendStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	blendStates[1].blendEnable = VK_FALSE;
	blendStates[1].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

	VkPipelineColorBlendStateCreateInfo blendInfo{};
	blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	blendInfo.logicOpEnable = VK_FALSE;
	blendInfo.attachmentCount = 2;
	blendInfo.pAttachments = blendStates;

	VkPipelineDepthStencilStateCreateInfo depthInfo{};
	depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	depthInfo.depthTestEnable = VK_TRUE;
	depthInfo.depthWriteEnable = VK_FALSE;
	depthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
	depthInfo.minDepthBounds = 0.0f;
	depthInfo.maxDepthBounds = 1.0f;

	VkGraphicsPipelineCreateInfo pipeInfo{};
	pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipeInfo.stageCount = 2;
	pipeInfo.pStages = stages;
	pipeInfo.pVertexInputState = &inputInfo;
	pipeInfo.pInputAssemblyState = &assemblyInfo;
	pipeInfo.pTessellationState = nullptr;
	pipeInfo.pViewportState = &viewportInfo;
	pipeInfo.pRasterizationState = &rasterInfo;
	pipeInfo.pMultisampleState = &multisampleInfo;
	pipeInfo.pDepthStencilState = &depthInfo;
	pipeInfo.pColorBlendState = &blendInfo;
	pipeInfo.pDynamicState = nullptr;
	pipeInfo.layout = this->pipelineLayout->getHandle();
	pipeInfo.renderPass = this->renderPass->getRenderPassHandle();
	pipeInfo.subpass = 0;

	VkPipeline pipe = VK_NULL_HANDLE;
	if (const auto res = vkCreateGraphicsPipelines(this->window->getDevice()->getDevice(), VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipe); VK_SUCCESS != res)
		throw Utils::Error("Unable to create graphics pipeline\n vkCreateGraphicsPipeline() returned %s\n", Utils::toString(res).c_str());

	this->pipeline = vk::Pipeline(this->window->getDevice()->getDevice(), pipe);
}