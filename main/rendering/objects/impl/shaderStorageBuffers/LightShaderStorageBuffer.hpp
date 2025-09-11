#pragma once

#include "../../base/ShaderStorageBuffer.hpp"
#include "../../../Uniforms.hpp"

class LightShaderStorageBuffer : public ShaderStorageBuffer {
public:
	LightShaderStorageBuffer(VulkanContext* context, std::vector<glsl::Light>* lightsUniform);

	void update();
private:
	std::vector<glsl::Light>* ssboData;
};