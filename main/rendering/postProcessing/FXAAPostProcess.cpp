#include "FXAAPostProcess.hpp"

#include "../RendererUtils.hpp"
#include "../objects/impl/framebuffers/WriteToTargetFramebuffer.hpp"
#include "../Driver.hpp"

FXAAPostProcess::FXAAPostProcess(Renderer* renderer) : PostProcessingEffect(renderer) {
	this->renderPass = this->renderer->getRenderPass("postProcessLDR");
	this->pipelineLayout = this->renderer->getPipelineLayout("mosaic");
	this->pipeline = this->renderer->getPipeline("fxaa");
}

TextureBuffer* FXAAPostProcess::apply(WriteToFramebufferPair framebuffers, std::uint32_t imageIndex, VkDescriptorSetPair readImages) {
	this->renderer->getDriver()->getTimestampManager().writeGPUTimestamp("fxaa", VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);

	RendererUtils::beginRenderPass(this->renderPass, framebuffers.second, imageIndex);
	RendererUtils::bindGraphicPipeline(this->pipeline->getHandle());
	RendererUtils::bindGraphicDescriptorSets(this->pipelineLayout->getHandle(), 0, 1, &readImages.second, 0, nullptr);
	RendererUtils::drawDirect(3, 1, 0, 0);
	RendererUtils::endRenderPass();

	this->renderer->getDriver()->getTimestampManager().writeGPUTimestamp("fxaa", VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

	return framebuffers.second->getRenderTarget();
}
