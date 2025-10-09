#include "PreSSAOPipeline.hpp"

#include "Error.hpp"
#include "toString.hpp"
#include "../../../../vulkan/VulkanDevice.hpp"
#include "../../../../vulkan/Swapchain.hpp"
#include "../../../PipelineCreation.hpp"

PreSSAOPipeline::PreSSAOPipeline(
	VulkanWindow* window,
	PipelineLayout* pipelineLayout,
	RenderPass* renderPass
) : Pipeline(window) {
	this->pipelineLayout = pipelineLayout;
	this->renderPass = renderPass;

	this->renderExtent = &this->window->getSwapchain()->getExtent();

	this->recreate();
}

void PreSSAOPipeline::recreate() {
	vk::ShaderModule vert = loadShaderModule(*this->window->getDevice(), "assets/main/shaders/forward.vert.spv");
	vk::ShaderModule frag = loadShaderModule(*this->window->getDevice(), "assets/main/shaders/preSSAO.frag.spv");

	VkPipelineShaderStageCreateInfo stages[2]{};
	stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
	stages[0].module = vert.handle;
	stages[0].pName = "main";

	stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	stages[1].module = frag.handle;
	stages[1].pName = "main";

	VkVertexInputBindingDescription vertexInputs[4]{};
	// Positions
	vertexInputs[0].binding = 0;
	vertexInputs[0].stride = sizeof(float) * 3;
	vertexInputs[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
	// UV
	vertexInputs[1].binding = 1;
	vertexInputs[1].stride = sizeof(float) * 2;
	vertexInputs[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
	// Fallback normals
	vertexInputs[2].binding = 2;
	vertexInputs[2].stride = sizeof(float) * 3;
	vertexInputs[2].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
	// TBN frame
	vertexInputs[3].binding = 3;
	vertexInputs[3].stride = sizeof(std::uint32_t);
	vertexInputs[3].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

	VkVertexInputAttributeDescription vertexAttributes[4]{};
	vertexAttributes[0].binding = 0;
	vertexAttributes[0].location = 0;
	vertexAttributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
	vertexAttributes[0].offset = 0;
	vertexAttributes[1].binding = 1;
	vertexAttributes[1].location = 1;
	vertexAttributes[1].format = VK_FORMAT_R32G32_SFLOAT;
	vertexAttributes[1].offset = 0;
	vertexAttributes[2].binding = 2;
	vertexAttributes[2].location = 2;
	vertexAttributes[2].format = VK_FORMAT_R32G32B32_SFLOAT;
	vertexAttributes[2].offset = 0;
	vertexAttributes[3].binding = 3;
	vertexAttributes[3].location = 3;
	vertexAttributes[3].format = VK_FORMAT_A2R10G10B10_UNORM_PACK32;
	vertexAttributes[3].offset = 0;

	VkPipelineVertexInputStateCreateInfo inputInfo{};
	inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	inputInfo.vertexBindingDescriptionCount = 4;
	inputInfo.pVertexBindingDescriptions = vertexInputs;
	inputInfo.vertexAttributeDescriptionCount = 4;
	inputInfo.pVertexAttributeDescriptions = vertexAttributes;

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
	rasterInfo.cullMode = VK_CULL_MODE_BACK_BIT;
	rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	rasterInfo.depthBiasEnable = VK_FALSE;
	rasterInfo.lineWidth = 1.0f;

	VkPipelineMultisampleStateCreateInfo multisampleInfo{};
	multisampleInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisampleInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

	VkPipelineColorBlendAttachmentState blendStates[1]{};
	blendStates[0].blendEnable = VK_FALSE;
	blendStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

	VkPipelineColorBlendStateCreateInfo blendInfo{};
	blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	blendInfo.logicOpEnable = VK_FALSE;
	blendInfo.attachmentCount = 1;
	blendInfo.pAttachments = blendStates;

	VkPipelineDepthStencilStateCreateInfo depthInfo{};
	depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	depthInfo.depthTestEnable = VK_TRUE;
	depthInfo.depthWriteEnable = VK_TRUE;
	depthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
	depthInfo.minDepthBounds = 0.0f;
	depthInfo.maxDepthBounds = 1.0f;

	VkDynamicState dynamicStates[1] = {
		VK_DYNAMIC_STATE_CULL_MODE
	};

	VkPipelineDynamicStateCreateInfo dynamicInfo{};
	dynamicInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamicInfo.dynamicStateCount = 1;
	dynamicInfo.pDynamicStates = dynamicStates;

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
	pipeInfo.pDynamicState = &dynamicInfo;
	pipeInfo.layout = this->pipelineLayout->getHandle();
	pipeInfo.renderPass = this->renderPass->getRenderPassHandle();
	pipeInfo.subpass = 0;

	VkPipeline pipe = VK_NULL_HANDLE;
	if (const auto res = vkCreateGraphicsPipelines(this->window->getDevice()->getDevice(), VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipe); VK_SUCCESS != res)
		throw Utils::Error("Unable to create graphics pipeline\n vkCreateGraphicsPipeline() returned %s\n", Utils::toString(res).c_str());

	this->pipeline = vk::Pipeline(this->window->getDevice()->getDevice(), pipe);
}