#include "DebugVisualisations.hpp"

#include "../Renderer.hpp"
#include "../RendererUtils.hpp"

namespace Debug {

	void renderDebugLightVolumes(Renderer* renderer, uint32_t imageIndex) {
		RenderPass* renderPass = renderer->getRenderPass("debugShapes");
		Framebuffer* framebuffer = renderer->getFramebuffer("debugShapes");
		Pipeline* pipeline = renderer->getPipeline("debugShapes");
		PipelineLayout* pipelineLayout = renderer->getPipelineLayout("lineDebug");

		VkDescriptorSet mvpDescriptorSet = RendererUtils::getDescriptorSetHandle(renderer->getDescriptorSet("mvp"));

		VkUtils::beginCmdLabel(RendererUtils::getCommandBuffer(), "Debug Light Volumes");
		RendererUtils::beginRenderPass(renderPass, framebuffer, imageIndex);
		RendererUtils::bindGraphicPipeline(pipeline->getHandle());
		RendererUtils::bindGraphicDescriptorSets(pipelineLayout->getHandle(), 0, 1, &mvpDescriptorSet);
		RendererUtils::drawMesh(renderer->getDebugSphere());
		RendererUtils::endRenderPass();
		VkUtils::endCmdLabel(RendererUtils::getCommandBuffer());
	}

}