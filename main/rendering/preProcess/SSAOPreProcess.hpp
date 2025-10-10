#pragma once

#include "../objects/base/PreProcessingEffect.hpp"

class DescriptorSet;

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

	DescriptorSet* mvpDescriptorSet = nullptr;

	DescriptorSet* projectionsUniformDescriptor = nullptr;
	DescriptorSet* ssaoUniformDescriptor = nullptr;
	DescriptorSet* ssaoTexturesDescriptor = nullptr;

	IUniformBuffer* projectionsUniformBuffer = nullptr;
	IUniformBuffer* ssaoUniformBuffer = nullptr;
};