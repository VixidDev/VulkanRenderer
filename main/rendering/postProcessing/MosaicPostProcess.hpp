#pragma once

#include "../objects/base/PostProcessingEffect.hpp"

class Renderer;

class MosaicPostProcess : public PostProcessingEffect {
public:
	MosaicPostProcess(Renderer* renderer);

	void recreate() override;
	void apply() override;
private:
};