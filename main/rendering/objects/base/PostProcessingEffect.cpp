#include "PostProcessingEffect.hpp"

PostProcessingEffect::PostProcessingEffect(Renderer* renderer) : renderer(renderer) {}

TextureBuffer* PostProcessingEffect::apply(WriteToFramebufferPair framebuffers, std::uint32_t imageIndex, VkDescriptorSetPair readImages) {
	return nullptr;
}

bool& PostProcessingEffect::getEnabled() {
	return this->enabled;
}