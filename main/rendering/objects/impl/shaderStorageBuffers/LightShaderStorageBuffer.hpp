#pragma once

#include "../../base/ShaderStorageBuffer.hpp"
#include "../../../Uniforms.hpp"

class LightShaderStorageBuffer : public ShaderStorageBuffer {
public:
	LightShaderStorageBuffer(VulkanContext* context, std::vector<glsl::Light>* lights);

	void update(VkCommandBuffer cmdBuff = VK_NULL_HANDLE) override;
private:
	std::vector<glsl::Light>* ssboData;
};