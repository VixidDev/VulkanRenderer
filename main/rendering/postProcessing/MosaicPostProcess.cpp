#include "MosaicPostProcess.hpp"

#include "../RendererUtils.hpp"
#include "../Driver.hpp"

MosaicPostProcess::MosaicPostProcess(Renderer* renderer) : PostProcessingEffect(renderer) {
	this->renderPass = this->renderer->getRenderPass("postProcess");
	this->pipelineLayout = this->renderer->getPipelineLayout("mosaic");
	this->pipeline = this->renderer->getPipeline("mosaic");
}

void MosaicPostProcess::apply(Framebuffer* framebuffer, std::uint32_t imageIndex, VkDescriptorSet readImage) {
	this->renderer->getDriver()->getTimestampManager().writeGPUTimestamp("mosaic", VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);

	RendererUtils::beginRenderPass(this->renderPass, framebuffer, imageIndex);
	RendererUtils::bindGraphicPipeline(this->pipeline->getHandle());
	RendererUtils::bindGraphicDescriptorSets(this->pipelineLayout->getHandle(), 0, 1, &readImage, 0, nullptr);
	RendererUtils::drawDirect(3, 1, 0, 0);
	RendererUtils::endRenderPass();

	this->renderer->getDriver()->getTimestampManager().writeGPUTimestamp("mosaic", VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
}
