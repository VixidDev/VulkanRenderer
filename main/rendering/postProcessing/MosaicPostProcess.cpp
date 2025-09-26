#include "MosaicPostProcess.hpp"

#include "../RendererUtils.hpp"

MosaicPostProcess::MosaicPostProcess(Renderer* renderer) : PostProcessingEffect(renderer) {
	this->renderPass = this->renderer->getRenderPass("postProcess");
	this->pipelineLayout = this->renderer->getPipelineLayout("singleImageSample");
	this->pipeline = this->renderer->getPipeline("mosaic");
}

void MosaicPostProcess::recreate() {}

void MosaicPostProcess::apply() {}
