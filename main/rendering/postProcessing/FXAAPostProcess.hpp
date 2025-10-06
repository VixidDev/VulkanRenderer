#pragma once

#include "../objects/base/PostProcessingEffect.hpp"

class FXAAPostProcess : public PostProcessingEffect {
public:
	FXAAPostProcess(Renderer* renderer);

	TextureBuffer* apply(WriteToFramebufferPair framebuffers, std::uint32_t imageIndex, VkDescriptorSetPair readImages) override;
private:
	RenderPass* renderPass = nullptr;
	PipelineLayout* pipelineLayout = nullptr;
	Pipeline* pipeline = nullptr;
};