#pragma once

#include "../objects/base/PreProcessingEffect.hpp"

class DescriptorSet;

struct SSAOBlurPC {
	int direction = 0;
	int radius = 2;
	float depthThreshold = 0.002f;
	float normalThreshold = 0.85f;
};

class SSAOPreProcess : public PreProcessingEffect {
public:
	SSAOPreProcess(Renderer* renderer);

	void apply(std::uint32_t imageIndex, bool needsPreSSAO = false);

	SSAOBlurPC blurPC{};
private:
	RenderPass* preRenderPass = nullptr;
	RenderPass* renderPass = nullptr;

	Framebuffer* preFramebuffer = nullptr;
	Framebuffer* framebuffer = nullptr;
	Framebuffer* blurHFramebuffer = nullptr;
	Framebuffer* blurVFramebuffer = nullptr;

	Pipeline* prePipeline = nullptr;
	Pipeline* pipeline = nullptr;
	Pipeline* blurPipeline = nullptr;

	PipelineLayout* prePipelineLayout = nullptr;
	PipelineLayout* pipelineLayout = nullptr;
	PipelineLayout* blurPipelineLayout = nullptr;

	DescriptorSet* mvpDescriptorSet = nullptr;

	DescriptorSet* projectionsUniformDescriptor = nullptr;
	DescriptorSet* ssaoUniformDescriptor = nullptr;
	DescriptorSet* ssaoTexturesDescriptor = nullptr;

	DescriptorSet* blurHDescriptor = nullptr;
	DescriptorSet* blurVDescriptor = nullptr;
	DescriptorSet* cameraPlanesDescriptor = nullptr;

	IUniformBuffer* projectionsUniformBuffer = nullptr;
	IUniformBuffer* ssaoUniformBuffer = nullptr;
	IUniformBuffer* cameraPlanesUniformBuffer = nullptr;
};