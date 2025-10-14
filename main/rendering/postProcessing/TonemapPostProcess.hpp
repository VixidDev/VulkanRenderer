#pragma once

#include "../objects/base/PostProcessingEffect.hpp"

// Should be in same order as in tonemap.frag
enum Tonemap {
	JUST_GAMMA,
	FILMIC,
	UNCHARTED,
	ACES,
	AGX,
	KHRONOS_PBR
};

struct TonemapPC {
	int tonemapType = Tonemap::JUST_GAMMA;
	float exposure = 2.0f;
};

class TonemapPostProcess : public PostProcessingEffect {
public:
	TonemapPostProcess(Renderer* renderer);

	TextureBuffer* apply(WriteToFramebufferPair framebuffers, std::uint32_t imageIndex, VkDescriptorSetPair readImages) override;

	int& getTonemap();
	float& getExposure();
private:
	RenderPass* renderPass = nullptr;
	PipelineLayout* pipelineLayout = nullptr;
	Pipeline* pipeline = nullptr;

	TonemapPC tonemapPushConst{};
};