#pragma once

#include "../objects/base/PostProcessingEffect.hpp"

class Renderer;
class DescriptorSet;
class Framebuffer;

class BloomPostProcess : public PostProcessingEffect {
public:
	BloomPostProcess(Renderer* renderer);

	TextureBuffer* apply(WriteToFramebufferPair framebuffers, std::uint32_t imageIndex, VkDescriptorSetPair readImages) override;
private:
	RenderPass* renderPass = nullptr;

	PipelineLayout* blurPipelineLayout = nullptr;
	PipelineLayout* compositionPipelineLayout = nullptr;

	Pipeline* blurPipeline = nullptr;
	Pipeline* compositionPipeline= nullptr;

	Framebuffer* intermediateFramebuffer = nullptr;
	Framebuffer* blurFramebuffer = nullptr;

	DescriptorSet* brightnessOutput = nullptr;
	DescriptorSet* intermediateOutput = nullptr;
	DescriptorSet* blurOutput = nullptr;
};