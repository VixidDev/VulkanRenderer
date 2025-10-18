#include "SSAOPreProcess.hpp"

#include "../RendererUtils.hpp"
#include "../Driver.hpp"

SSAOPreProcess::SSAOPreProcess(Renderer* renderer) : PreProcessingEffect(renderer) {
	this->preRenderPass = this->renderer->getRenderPass("pre_ssao");
	this->renderPass = this->renderer->getRenderPass("ssao");

	this->preFramebuffer = this->renderer->getFramebuffer("pre_ssao");
	this->framebuffer = this->renderer->getFramebuffer("ssao");
	this->blurHFramebuffer = this->renderer->getFramebuffer("ssaoHblur"); // Takes ssao, writes to ssaoHblur
	this->blurVFramebuffer = this->renderer->getFramebuffer("ssaoVblur"); // Takes ssaoHblur, writes to ssaoVblur

	this->prePipeline = this->renderer->getPipeline("pre_ssao");
	this->pipeline = this->renderer->getPipeline("ssao");
	this->blurPipeline = this->renderer->getPipeline("ssao_blur");

	this->prePipelineLayout = this->renderer->getPipelineLayout("pre_ssao");
	this->pipelineLayout = this->renderer->getPipelineLayout("ssao");
	this->blurPipelineLayout = this->renderer->getPipelineLayout("ssao_blur");

	this->mvpDescriptorSet = this->renderer->getDescriptorSet("mvp");
	this->projectionsUniformDescriptor = this->renderer->getDescriptorSet("projections");
	this->ssaoUniformDescriptor = this->renderer->getDescriptorSet("ssao");
	this->ssaoTexturesDescriptor = this->renderer->getDescriptorSet("ssaoTextures");
	this->blurHDescriptor = this->renderer->getDescriptorSet("ssaoHBlurTextures");
	this->blurVDescriptor = this->renderer->getDescriptorSet("ssaoVBlurTextures");
	this->cameraPlanesDescriptor = this->renderer->getDescriptorSet("cameraPlanes");

	this->projectionsUniformBuffer = this->renderer->getUniformBuffer("projections");
	this->ssaoUniformBuffer = this->renderer->getUniformBuffer("ssao");
	this->cameraPlanesUniformBuffer = this->renderer->getUniformBuffer("cameraPlanes");
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
			&RendererUtils::getDescriptorSetHandle(this->mvpDescriptorSet));

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
	std::vector<VkDescriptorSet> descriptorSets = { 
		RendererUtils::getDescriptorSetHandle(this->projectionsUniformDescriptor), 
		RendererUtils::getDescriptorSetHandle(this->ssaoUniformDescriptor), 
		RendererUtils::getDescriptorSetHandle(this->ssaoTexturesDescriptor) };
	RendererUtils::bindGraphicDescriptorSets(this->pipelineLayout->getHandle(), 0, descriptorSets.size(), descriptorSets.data());
	RendererUtils::drawDirect(3, 1, 0, 0);
	RendererUtils::endRenderPass();

	this->renderer->getDriver()->getTimestampManager().writeGPUTimestamp("ssao", VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

	// SSAO blur using a bilateral (edge-preserving) filter
	this->renderer->getDriver()->getTimestampManager().writeGPUTimestamp("ssaoBlur", VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);

	RendererUtils::updateUniformBuffer(this->cameraPlanesUniformBuffer);

	this->blurPC.direction = 0;

	// Horizontal pass
	RendererUtils::beginRenderPass(this->renderPass, this->blurHFramebuffer, imageIndex);
	RendererUtils::bindGraphicPipeline(this->blurPipeline->getHandle());
	RendererUtils::bindPushConstant(this->blurPipelineLayout->getHandle(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(SSAOBlurPC), &this->blurPC);
	RendererUtils::bindGraphicDescriptorSets(this->blurPipelineLayout->getHandle(), 0, 1, &RendererUtils::getDescriptorSetHandle(this->blurHDescriptor));
	RendererUtils::bindGraphicDescriptorSets(this->blurPipelineLayout->getHandle(), 1, 1, &RendererUtils::getDescriptorSetHandle(this->cameraPlanesDescriptor));
	RendererUtils::drawDirect(3, 1, 0, 0);
	RendererUtils::endRenderPass();

	this->blurPC.direction = 1;

	// Vertical pass
	RendererUtils::beginRenderPass(this->renderPass, this->blurVFramebuffer, imageIndex);
	RendererUtils::bindGraphicPipeline(this->blurPipeline->getHandle());
	RendererUtils::bindPushConstant(this->blurPipelineLayout->getHandle(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(SSAOBlurPC), &this->blurPC);
	RendererUtils::bindGraphicDescriptorSets(this->blurPipelineLayout->getHandle(), 0, 1, &RendererUtils::getDescriptorSetHandle(this->blurVDescriptor));
	RendererUtils::bindGraphicDescriptorSets(this->blurPipelineLayout->getHandle(), 1, 1, &RendererUtils::getDescriptorSetHandle(this->cameraPlanesDescriptor));
	RendererUtils::drawDirect(3, 1, 0, 0);
	RendererUtils::endRenderPass();

	this->renderer->getDriver()->getTimestampManager().writeGPUTimestamp("ssaoBlur", VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
}
