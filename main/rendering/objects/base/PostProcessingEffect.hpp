#pragma once

#include "../vulkan/objects/VkObjects.hpp"

class Renderer;
class TextureBuffer;
class WriteToTargetFramebuffer;
class RenderPass;
class PipelineLayout;
class Pipeline;

using VkDescriptorSetPair = std::pair<VkDescriptorSet, VkDescriptorSet>;
using WriteToFramebufferPair = std::pair<WriteToTargetFramebuffer*, WriteToTargetFramebuffer*>;

class PostProcessingEffect {
public:
	PostProcessingEffect() = default;
	PostProcessingEffect(Renderer* renderer);

	virtual TextureBuffer* apply(WriteToFramebufferPair framebuffers, std::uint32_t imageIndex, VkDescriptorSetPair readImages);

	bool& getEnabled();
protected:
	Renderer* renderer;
	bool enabled = false;
};