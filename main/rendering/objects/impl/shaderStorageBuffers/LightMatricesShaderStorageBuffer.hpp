#pragma once

#include "../../base/ShaderStorageBuffer.hpp"
#include "../../../Uniforms.hpp"

class LightMatricesShaderStorageBuffer : public ShaderStorageBuffer {
public:
	LightMatricesShaderStorageBuffer(VulkanContext* context, std::vector<glm::mat4>* lightMatrices);

	void update();
private:
	std::vector<glm::mat4>* ssboData;
};