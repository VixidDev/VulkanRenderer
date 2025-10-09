#include "PreProcessingEffect.hpp"

PreProcessingEffect::PreProcessingEffect(Renderer* renderer) : renderer(renderer) {}

void PreProcessingEffect::apply(std::uint32_t imageIndex) {}

bool& PreProcessingEffect::getEnabled() {
	return this->enabled;
}