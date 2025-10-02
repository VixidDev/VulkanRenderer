#include "BloomPostProcess.hpp"

#include "../RendererUtils.hpp"
#include "../objects/base/DescriptorSet.hpp"
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
	this->intermediateFramebuffer = this->renderer->getFramebuffer("writeToIntermediate2");
	this->blurFramebuffer = this->renderer->getFramebuffer("writeToBlur");

	// Descriptor sets for binding samplers
	this->brightnessOutput = this->renderer->getDescriptorSet("brightness");
	this->intermediate2Output = this->renderer->getDescriptorSet("intermediate2");
	this->blurOutput = this->renderer->getDescriptorSet("blurOutput");
}

void BloomPostProcess::apply(Framebuffer* framebuffer, std::uint32_t imageIndex, VkDescriptorSet readImage) {
	// Need to ping-pong between 2 framebuffers for horizontal and vertical blur passes

	this->renderer->getDriver()->getTimestampManager().writeGPUTimestamp("bloom", VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);

	// Framebuffers to ping-pong
	Framebuffer* framebuffer1 = this->intermediateFramebuffer;
	Framebuffer* framebuffer2 = this->blurFramebuffer;

	// Render targets
	VkDescriptorSet renderTarget1 = this->blurOutput->getHandle();
	VkDescriptorSet renderTarget2 = this->intermediate2Output->getHandle();

	// i = 0 - read: brightness   - write: intermediate
	// i = 1 - read: intermediate - write: blur        
	// i = 2 - read: blur		  - write: intermediate
	// i = 3 - read: intermediate - write: blur        
	// i = 4 - read: blur		  - write: intermediate
	// i = 5 - read: intermediate - write: blur        
	// i = 6 - read: blur		  - write: intermediate
	// i = 7 - read: intermediate - write: blur        
	// i = 8 - read: blur		  - write: intermediate
	// i = 9 - read: intermediate - write: blur        

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

	// Write to 'intermediate' using 'sceneOutput' and 'brightness' as input
	// (Could just make a composition post process and directly call that with the input images)
	RendererUtils::beginRenderPass(this->renderPass, framebuffer, imageIndex);
	RendererUtils::bindGraphicPipeline(this->compositionPipeline->getHandle());
	RendererUtils::bindGraphicDescriptorSets(this->compositionPipelineLayout->getHandle(), 0, 1, &readImage, 0, nullptr); // scene image
	RendererUtils::bindGraphicDescriptorSets(this->compositionPipelineLayout->getHandle(), 1, 1, &this->blurOutput->getHandle(), 0, nullptr); // blur output
	RendererUtils::drawDirect(3, 1, 0, 0);
	RendererUtils::endRenderPass();

	this->renderer->getDriver()->getTimestampManager().writeGPUTimestamp("bloom", VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
}
