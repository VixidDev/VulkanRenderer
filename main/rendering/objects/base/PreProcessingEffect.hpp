#pragma once

#include "../vulkan/objects/VkObjects.hpp"

class Renderer;
class RenderPass;
class PipelineLayout;
class Pipeline;
class Framebuffer;
class IUniformBuffer;

class PreProcessingEffect {
public:
	PreProcessingEffect() = default;
	PreProcessingEffect(Renderer* renderer);

	virtual void apply(std::uint32_t imageIndex);

	bool& getEnabled();
protected:
	Renderer* renderer;
	bool enabled = false;
};