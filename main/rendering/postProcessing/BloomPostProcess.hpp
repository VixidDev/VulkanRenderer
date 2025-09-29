#pragma once

#include "../objects/base/PostProcessingEffect.hpp"

class Renderer;
class DescriptorSet;

class BloomPostProcess : public PostProcessingEffect {
public:
	BloomPostProcess(Renderer* renderer);

	void apply(Framebuffer* framebuffer, std::uint32_t imageIndex, VkDescriptorSet readImage) override;
private:
	RenderPass* renderPass = nullptr;

	PipelineLayout* blurPipelineLayout = nullptr;
	PipelineLayout* compositionPipelineLayout = nullptr;

	Pipeline* blurPipeline = nullptr;
	Pipeline* compositionPipeline= nullptr;

	Framebuffer* intermediateFramebuffer = nullptr;
	Framebuffer* blurFramebuffer = nullptr;

	DescriptorSet* brightnessOutput;
	DescriptorSet* intermediate2Output;
	DescriptorSet* blurOutput;
};