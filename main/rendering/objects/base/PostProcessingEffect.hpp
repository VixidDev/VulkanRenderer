#pragma once

class Renderer;
class RenderPass;
class PipelineLayout;
class Pipeline;

class PostProcessingEffect {
public:
	PostProcessingEffect() = default;
	PostProcessingEffect(Renderer* renderer);

	virtual void recreate();
	virtual void apply();

	RenderPass* getRenderPass();
	PipelineLayout* getPipelineLayout();
	Pipeline* getPipeline();

	bool& getEnabled();
protected:
	Renderer* renderer;
	bool enabled = false;

	RenderPass* renderPass = nullptr;
	PipelineLayout* pipelineLayout = nullptr;
	Pipeline* pipeline = nullptr;
};