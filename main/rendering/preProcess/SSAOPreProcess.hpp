#pragma once

#include "../objects/base/PreProcessingEffect.hpp"

class SSAOPreProcess : public PreProcessingEffect {
public:
	SSAOPreProcess(Renderer* renderer);

	void apply(std::uint32_t imageIndex, bool needsPreSSAO = false);
private:
	RenderPass* preRenderPass = nullptr;
	RenderPass* renderPass = nullptr;

	Framebuffer* preFramebuffer = nullptr;
	Framebuffer* framebuffer = nullptr;

	Pipeline* prePipeline = nullptr;
	Pipeline* pipeline = nullptr;

	PipelineLayout* prePipelineLayout = nullptr;
	PipelineLayout* pipelineLayout = nullptr;

	VkDescriptorSet mvpDescriptorSet = VK_NULL_HANDLE;
	VkDescriptorSet projectionsUniformDescriptor = VK_NULL_HANDLE;
	VkDescriptorSet ssaoUniformDescriptor = VK_NULL_HANDLE;
	VkDescriptorSet ssaoTexturesDescriptor = VK_NULL_HANDLE;

	IUniformBuffer* projectionsUniformBuffer = nullptr;
	IUniformBuffer* ssaoUniformBuffer = nullptr;
};