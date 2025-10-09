#include "SSAOPreProcess.hpp"

#include "../RendererUtils.hpp"
#include "../Driver.hpp"

SSAOPreProcess::SSAOPreProcess(Renderer* renderer) : PreProcessingEffect(renderer) {
	this->preRenderPass = this->renderer->getRenderPass("pre_ssao");
	this->renderPass = this->renderer->getRenderPass("ssao");

	this->preFramebuffer = this->renderer->getFramebuffer("pre_ssao");
	this->framebuffer = this->renderer->getFramebuffer("ssao");

	this->prePipeline = this->renderer->getPipeline("pre_ssao");
	this->pipeline = this->renderer->getPipeline("ssao");

	this->prePipelineLayout = this->renderer->getPipelineLayout("deferredWriting");
	this->pipelineLayout = this->renderer->getPipelineLayout("ssao");

	this->mvpDescriptorSet = this->renderer->getDescriptorSet("mvp")->getHandle();
	this->projectionsUniformDescriptor = this->renderer->getDescriptorSet("projections")->getHandle();
	this->ssaoUniformDescriptor = this->renderer->getDescriptorSet("ssao")->getHandle();
	this->ssaoTexturesDescriptor = this->renderer->getDescriptorSet("ssaoTextures")->getHandle();

	this->projectionsUniformBuffer = this->renderer->getUniformBuffer("projections");
	this->ssaoUniformBuffer = this->renderer->getUniformBuffer("ssao");
}

void SSAOPreProcess::apply(std::uint32_t imageIndex, bool needsPreSSAO) {
	// Pre SSAO is a step to fill normals gbuffer and depth buffer
	// needed to perform SSAO, if doing deferred this is not needed
	if (needsPreSSAO) {
		std::vector<MeshData>& meshData = this->renderer->getDriver()->getMeshData();

		this->renderer->getDriver()->getTimestampManager().writeGPUTimestamp("pre_ssao", VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);

		RendererUtils::beginRenderPass(this->preRenderPass, this->preFramebuffer, imageIndex);
		RendererUtils::bindGraphicPipeline(this->prePipeline->getHandle());
		RendererUtils::bindGraphicDescriptorSets(
			this->prePipelineLayout->getHandle(), 0, 1,
			&this->mvpDescriptorSet);

		auto perMeshCallback = [this](MeshData& meshData) {
			RendererUtils::bindGraphicDescriptorSets(
				this->prePipelineLayout->getHandle(), 1, 1,
				&this->renderer->getDriver()->getMaterialDescriptors().at(meshData.materialId));
		};

		RendererUtils::setCullMode(VK_CULL_MODE_BACK_BIT);

		// Draw non-alpha masked meshes
		for (std::size_t i = 0; i < meshData.size(); i++) {
			if (meshData[i].hasAlphaMask) continue;

			RendererUtils::drawMesh(meshData[i], perMeshCallback);
		}

		RendererUtils::setCullMode(VK_CULL_MODE_NONE);

		// Draw alpha masked meshes
		for (std::uint32_t i = 0; i < meshData.size(); i++) {
			if (!meshData[i].hasAlphaMask) continue;

			RendererUtils::drawMesh(meshData[i], perMeshCallback);
		}

		RendererUtils::endRenderPass();

		this->renderer->getDriver()->getTimestampManager().writeGPUTimestamp("pre_ssao", VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
	}

	// Regular SSAO pass
	this->renderer->getDriver()->getTimestampManager().writeGPUTimestamp("ssao", VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);

	RendererUtils::updateUniformBuffer(this->projectionsUniformBuffer);
	// TODO: These values never change and so don't need to constantly be updated every frame!
	RendererUtils::updateUniformBuffer(this->ssaoUniformBuffer);

	RendererUtils::beginRenderPass(this->renderPass, this->framebuffer, imageIndex);
	RendererUtils::bindGraphicPipeline(this->pipeline->getHandle());
	std::vector<VkDescriptorSet> descriptorSets = { this->projectionsUniformDescriptor, this->ssaoUniformDescriptor, this->ssaoTexturesDescriptor };
	RendererUtils::bindGraphicDescriptorSets(this->pipelineLayout->getHandle(), 0, descriptorSets.size(), descriptorSets.data());
	RendererUtils::drawDirect(3, 1, 0, 0);
	RendererUtils::endRenderPass();

	this->renderer->getDriver()->getTimestampManager().writeGPUTimestamp("ssao", VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
}