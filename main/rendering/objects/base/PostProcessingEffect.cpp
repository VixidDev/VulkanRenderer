#include "PostProcessingEffect.hpp"

PostProcessingEffect::PostProcessingEffect(Renderer* renderer) : renderer(renderer) {}

void PostProcessingEffect::recreate() {}

void PostProcessingEffect::apply() {}

RenderPass* PostProcessingEffect::getRenderPass() {
	return this->renderPass;
}

PipelineLayout* PostProcessingEffect::getPipelineLayout() {
	return this->pipelineLayout;
}

Pipeline* PostProcessingEffect::getPipeline() {
	return this->pipeline;
}

bool& PostProcessingEffect::getEnabled() {
	return this->enabled;
}