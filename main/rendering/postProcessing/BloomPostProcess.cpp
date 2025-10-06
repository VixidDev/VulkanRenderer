#include "BloomPostProcess.hpp"

#include "../RendererUtils.hpp"
#include "../objects/base/DescriptorSet.hpp"
#include "../objects/impl/framebuffers/WriteToTargetFramebuffer.hpp"
#include "../Driver.hpp"

BloomPostProcess::BloomPostProcess(Renderer* renderer) : PostProcessingEffect(renderer) {
	// Same for both steps
	this->renderPass = this->renderer->getRenderPass("postProcess");

	// Pipeline layouts for blur step then composition step
	this->blurPipelineLayout = this->renderer->getPipelineLayout("bloom");
	this->compositionPipelineLayout = this->renderer->getPipelineLayout("composition");

	// Pipelines for blur step then composition step
	this->blurPipeline = this->renderer->getPipeline("bloom");
	this->compositionPipeline = this->renderer->getPipeline("composition");

	// Framebuffers for writing to targets
	this->intermediateFramebuffer = this->renderer->getFramebuffer("writeToIntermediateHDR2");
	this->blurFramebuffer = this->renderer->getFramebuffer("writeToBlur");

	// Descriptor sets for binding samplers
	this->brightnessOutput = this->renderer->getDescriptorSet("brightness");
	this->intermediateOutput = this->renderer->getDescriptorSet("intermediateHDR2");
	this->blurOutput = this->renderer->getDescriptorSet("blurOutput");
}

TextureBuffer* BloomPostProcess::apply(WriteToFramebufferPair framebuffers, std::uint32_t imageIndex, VkDescriptorSetPair readImages) {
	// Need to ping-pong between 2 framebuffers for horizontal and vertical blur passes

	this->renderer->getDriver()->getTimestampManager().writeGPUTimestamp("bloom", VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);

	// Framebuffers to ping-pong
	Framebuffer* framebuffer1 = this->intermediateFramebuffer;
	Framebuffer* framebuffer2 = this->blurFramebuffer;

	// Render targets
	VkDescriptorSet renderTarget1 = this->blurOutput->getHandle();
	VkDescriptorSet renderTarget2 = this->intermediateOutput->getHandle();

	// Ping-pong example for 6 iterations (just to help visualise destination render targets)
	// i = 0 - read: brightness			- write: intermediate_hdr
	// i = 1 - read: intermediate_hdr	- write: blur        
	// i = 2 - read: blur				- write: intermediate_hdr
	// i = 3 - read: intermediate_hdr	- write: blur        
	// i = 4 - read: blur				- write: intermediate_hdr
	// i = 5 - read: intermediate_hdr	- write: blur           

	// In practice any value of bloomIterations over 1 costs way too much 
	// computation time for such little visual benefit gained from doing so
	int iterations = this->renderer->bloomIterations * 2; // Must be even
	bool firstPass = true;
	for (int i = 0, direction = 0; i < iterations; i++, direction = 1 - direction) {
		RendererUtils::beginRenderPass(this->renderPass, framebuffer1, imageIndex);
		RendererUtils::bindGraphicPipeline(this->blurPipeline->getHandle());
		VkDescriptorSet inputImage = firstPass ? this->brightnessOutput->getHandle() : renderTarget1;
		RendererUtils::bindGraphicDescriptorSets(this->blurPipelineLayout->getHandle(), 0, 1, &inputImage, 0, nullptr);
		RendererUtils::bindPushConstant(this->blurPipelineLayout->getHandle(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(int), &direction);
		RendererUtils::drawDirect(3, 1, 0, 0);
		RendererUtils::endRenderPass();

		std::swap(framebuffer1, framebuffer2);
		std::swap(renderTarget1, renderTarget2);

		if (firstPass) firstPass = false;
	}

	// Combine blurred image with scene image

	// Write to HDR intermediate using 'HDR' and 'brightness' as input
	// (Could just make a composition post process and directly call that with the input images)
	RendererUtils::beginRenderPass(this->renderPass, framebuffers.first, imageIndex);
	RendererUtils::bindGraphicPipeline(this->compositionPipeline->getHandle());
	RendererUtils::bindGraphicDescriptorSets(this->compositionPipelineLayout->getHandle(), 0, 1, &readImages.first, 0, nullptr); // scene image
	RendererUtils::bindGraphicDescriptorSets(this->compositionPipelineLayout->getHandle(), 1, 1, &this->blurOutput->getHandle(), 0, nullptr); // blur output
	RendererUtils::drawDirect(3, 1, 0, 0);
	RendererUtils::endRenderPass();

	this->renderer->getDriver()->getTimestampManager().writeGPUTimestamp("bloom", VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

	// Return last written to render target
	return framebuffers.first->getRenderTarget();
}
