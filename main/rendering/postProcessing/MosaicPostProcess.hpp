#pragma once

#include "../objects/base/PostProcessingEffect.hpp"

class MosaicPostProcess : public PostProcessingEffect {
public:
	MosaicPostProcess(Renderer* renderer);

	void apply(Framebuffer* framebuffer, std::uint32_t imageIndex, VkDescriptorSet readImage) override;
private:
	RenderPass* renderPass = nullptr;
	PipelineLayout* pipelineLayout = nullptr;
	Pipeline* pipeline = nullptr;
};