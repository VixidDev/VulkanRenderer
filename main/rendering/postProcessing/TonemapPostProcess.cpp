#include "TonemapPostProcess.hpp"

#include "../RendererUtils.hpp"
#include "../objects/impl/framebuffers/WriteToTargetFramebuffer.hpp"
#include "../Driver.hpp"

TonemapPostProcess::TonemapPostProcess(Renderer* renderer) : PostProcessingEffect(renderer) {
	this->enabled = true; // Tonemapping should always be on

	this->renderPass = this->renderer->getRenderPass("tonemap");
	this->pipelineLayout = this->renderer->getPipelineLayout("tonemap");
	this->pipeline = this->renderer->getPipeline("tonemap");
}

// Takes in HDR texture and tonemaps to LDR, converts to sRGB colour space and outputs
// Luma value in A channel for FXAA
TextureBuffer* TonemapPostProcess::apply(WriteToFramebufferPair framebuffers, std::uint32_t imageIndex, VkDescriptorSetPair readImages) {
	this->renderer->getDriver()->getTimestampManager().writeGPUTimestamp("tonemap", VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);

	RendererUtils::beginRenderPass(this->renderPass, framebuffers.second, imageIndex);
	RendererUtils::bindGraphicPipeline(this->pipeline->getHandle());
	RendererUtils::bindGraphicDescriptorSets(this->pipelineLayout->getHandle(), 0, 1, &readImages.first, 0, nullptr);
	RendererUtils::bindPushConstant(this->pipelineLayout->getHandle(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(TonemapPC), &this->tonemapPushConst); 
	RendererUtils::drawDirect(3, 1, 0, 0);
	RendererUtils::endRenderPass();

	this->renderer->getDriver()->getTimestampManager().writeGPUTimestamp("tonemap", VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

	return framebuffers.second->getRenderTarget();
}

int& TonemapPostProcess::getTonemap() {
	return this->tonemapPushConst.tonemapType;
}

float& TonemapPostProcess::getExposure() {
	return this->tonemapPushConst.exposure;
}
