#pragma once

#include <cstdint>

class Renderer;
class TextureBuffer;

namespace Debug {

	void renderDebugLightVolumes(Renderer* renderer, uint32_t imageIndex);

}