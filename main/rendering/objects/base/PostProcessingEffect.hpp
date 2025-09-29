#pragma once

#include "../vulkan/objects/VkObjects.hpp"

class Renderer;
class RenderPass;
class PipelineLayout;
class Pipeline;
class Framebuffer;

class PostProcessingEffect {
public:
	PostProcessingEffect() = default;
	PostProcessingEffect(Renderer* renderer);

	virtual void apply(Framebuffer* framebuffer, std::uint32_t imageIndex, VkDescriptorSet readImage);

	bool& getEnabled();
protected:
	Renderer* renderer;
	bool enabled = false;
};