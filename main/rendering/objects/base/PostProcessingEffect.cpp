#include "PostProcessingEffect.hpp"

PostProcessingEffect::PostProcessingEffect(Renderer* renderer) : renderer(renderer) {}

void PostProcessingEffect::apply(Framebuffer* framebuffer, std::uint32_t imageIndex, VkDescriptorSet readImage) {}

bool& PostProcessingEffect::getEnabled() {
	return this->enabled;
}