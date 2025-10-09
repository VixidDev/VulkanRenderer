#pragma once

#include "../../base/TextureBuffer.hpp"

class NoiseTextureBuffer : public TextureBuffer {
public:
	NoiseTextureBuffer(VulkanContext* context);

	void recreate();
private:
};