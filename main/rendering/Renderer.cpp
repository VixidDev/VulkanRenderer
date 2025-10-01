#include "Renderer.hpp"

#include <iostream>

#include "Error.hpp"
#include "toString.hpp"

#include "../Driver.hpp"
#include "../baked/BakedModel.hpp"
#include "../baked/BakedModelLoader.hpp"

#include "PipelineCreation.hpp"

#include "objects/impl/renderPasses/ForwardPass.hpp"
#include "objects/impl/renderPasses/DeferredPass.hpp"
#include "objects/impl/renderPasses/ShadowPass.hpp"
#include "objects/impl/renderPasses/GUIPass.hpp"
#include "objects/impl/renderPasses/SunViewPass.hpp"
#include "objects/impl/renderPasses/PostProcessPass.hpp"

#include "objects/impl/pipelineLayouts/ForwardPipelineLayout.hpp"
#include "objects/impl/pipelineLayouts/DeferredWritingPipelineLayout.hpp"
#include "objects/impl/pipelineLayouts/DeferredShadingPipelineLayout.hpp"
#include "objects/impl/pipelineLayouts/ShadowPipelineLayout.hpp"
#include "objects/impl/pipelineLayouts/CubemapShadowPipelineLayout.hpp"
#include "objects/impl/pipelineLayouts/LineDebugPipelineLayout.hpp"
#include "objects/impl/pipelineLayouts/DebugViewsPipelineLayout.hpp"
#include "objects/impl/pipelineLayouts/OverVisualisationPipelineLayout.hpp"
#include "objects/impl/pipelineLayouts/BloomPipelineLayout.hpp"
#include "objects/impl/pipelineLayouts/CompositionPipelineLayout.hpp"
#include "objects/impl/pipelineLayouts/MosaicPipelineLayout.hpp"

#include "objects/impl/pipelines/ForwardPipeline.hpp"
#include "objects/impl/pipelines/DeferredWritingPipeline.hpp"
#include "objects/impl/pipelines/DeferredShadingPipeline.hpp"
#include "objects/impl/pipelines/ShadowPipeline.hpp"
#include "objects/impl/pipelines/CubemapShadowPipeline.hpp"
#include "objects/impl/pipelines/LineDebugPipeline.hpp"
#include "objects/impl/pipelines/DebugViewsPipeline.hpp"
#include "objects/impl/pipelines/OverVisualisationPipeline.hpp"
#include "objects/impl/pipelines/MosaicPipeline.hpp"
#include "objects/impl/pipelines/BloomPipeline.hpp"
#include "objects/impl/pipelines/CompositionPipeline.hpp"

#include "objects/impl/textureBuffers/DepthTextureBuffer.hpp"
#include "objects/impl/textureBuffers/CubemapDepthTextureBuffer.hpp"
#include "objects/impl/textureBuffers/CubemapArrayDepthTextureBuffer.hpp"
#include "objects/impl/textureBuffers/ArrayColourTextureBuffer.hpp"
#include "objects/impl/textureBuffers/ArrayDepthTextureBuffer.hpp"
#include "objects/impl/textureBuffers/ColourTextureBuffer.hpp"

#include "objects/impl/framebuffers/ForwardFramebuffer.hpp"
#include "objects/impl/framebuffers/DeferredFramebuffer.hpp"
#include "objects/impl/framebuffers/ShadowFramebuffer.hpp"
#include "objects/impl/framebuffers/CubemapFramebuffer.hpp"
#include "objects/impl/framebuffers/CubemapArrayFramebuffer.hpp"
#include "objects/impl/framebuffers/ArrayFramebuffer.hpp"
#include "objects/impl/framebuffers/GUIFramebuffer.hpp"
#include "objects/impl/framebuffers/SunFramebuffer.hpp"
#include "objects/impl/framebuffers/OutputFramebuffer.hpp"
#include "objects/impl/framebuffers/IntermediateFramebuffer.hpp"
#include "objects/impl/framebuffers/Intermediate2Framebuffer.hpp"
#include "objects/impl/framebuffers/BlurFramebuffer.hpp"

#include "objects/impl/descriptorSets/BufferDescriptorSet.hpp"
#include "objects/impl/descriptorSets/ImageDescriptorSet.hpp"
#include "objects/impl/descriptorSets/ArrayImageDescriptorSet.hpp"

#include "postProcessing/MosaicPostProcess.hpp"
#include "postProcessing/BloomPostProcess.hpp"

#include "../vulkan/VulkanDevice.hpp"
#include "../vulkan/VkUtils.hpp"
#include "RendererUtils.hpp"

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

Renderer::Renderer(Driver* driver) : driver(driver) {
	this->context.window = initialiseVulkanWindow();
	this->context.allocator = initialiseVulkanAllocator(*this->context.window);

	this->createDummyTexture();

	VulkanWindow* window = this->context.window.get();
	VulkanAllocator* allocator = this->context.allocator.get();

	this->camera = Camera(window, 90.0f, 0.01f, 256.0f, glm::vec3(0.0f, 7.0f, -12.0f), glm::vec3(0.0f, 0.0f, -1.0f));

	// Render passes
	this->renderPasses.emplace("forward", std::make_unique<ForwardPass>(window, &this->sampleCountSetting));
	this->renderPasses.emplace("deferred", std::make_unique<DeferredPass>(window, &this->sampleCountSetting));
	this->renderPasses.emplace("shadow", std::make_unique<ShadowPass>(window, &this->sampleCountSetting));
	this->renderPasses.emplace("gui", std::make_unique<GUIPass>(window, &this->sampleCountSetting));
	this->renderPasses.emplace("sunView", std::make_unique<SunViewPass>(window, &this->sampleCountSetting));
	this->renderPasses.emplace("postProcess", std::make_unique<PostProcessPass>(window));

	// Descriptor Set Layouts
	std::vector<DescriptorSetting> uniformBufferV = { { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT } };
	std::vector<DescriptorSetting> uniformBufferF = { { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT } };
	std::vector<DescriptorSetting> uniformBufferVF = { { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT } };
	std::vector<DescriptorSetting> ssboF = { { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT } };
	std::vector<DescriptorSetting> imageF = { { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT } };
	std::vector<DescriptorSetting> materialSettings = {
		{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT },
		{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT },
		{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT },
		{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT },
		{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT },
		{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT } };
	std::vector<DescriptorSetting> deferredInputAttachments = {
		{ VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, VK_SHADER_STAGE_FRAGMENT_BIT },
		{ VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, VK_SHADER_STAGE_FRAGMENT_BIT },
		{ VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, VK_SHADER_STAGE_FRAGMENT_BIT },
		{ VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, VK_SHADER_STAGE_FRAGMENT_BIT } };

	this->descriptorSetLayouts.emplace("uboV", createDescriptorLayout(*window, uniformBufferV));
	this->descriptorSetLayouts.emplace("uboF", createDescriptorLayout(*window, uniformBufferF));
	this->descriptorSetLayouts.emplace("uboVF", createDescriptorLayout(*window, uniformBufferVF));
	this->descriptorSetLayouts.emplace("ssboF", createDescriptorLayout(*window, ssboF));
	this->descriptorSetLayouts.emplace("imageF", createDescriptorLayout(*window, imageF));
	this->descriptorSetLayouts.emplace("materials", createDescriptorLayout(*window, materialSettings));
	this->descriptorSetLayouts.emplace("deferredInputAttachments", createDescriptorLayout(*window, deferredInputAttachments));

	// Pipeline Layouts
	this->pipelineLayouts.emplace("shadow", std::make_unique<ShadowPipelineLayout>(window, &this->descriptorSetLayouts));
	this->pipelineLayouts.emplace("cubemapShadow", std::make_unique<CubemapShadowPipelineLayout>(window, &this->descriptorSetLayouts));
	this->pipelineLayouts.emplace("forward", std::make_unique<ForwardPipelineLayout>(window, &this->descriptorSetLayouts, &this->shadowsEnabled));
	this->pipelineLayouts.emplace("deferredWriting", std::make_unique<DeferredWritingPipelineLayout>(window, &this->descriptorSetLayouts));
	this->pipelineLayouts.emplace("deferredShading", std::make_unique<DeferredShadingPipelineLayout>(window, &this->descriptorSetLayouts, &this->shadowsEnabled));
	this->pipelineLayouts.emplace("bloom", std::make_unique<BloomPipelineLayout>(window, &this->descriptorSetLayouts));
	this->pipelineLayouts.emplace("mosaic", std::make_unique<MosaicPipelineLayout>(window, &this->descriptorSetLayouts));
	this->pipelineLayouts.emplace("composition", std::make_unique<CompositionPipelineLayout>(window, &this->descriptorSetLayouts));
	
	// Debug pipeline layouts
	this->pipelineLayouts.emplace("lineDebug", std::make_unique<LineDebugPipelineLayout>(window, &this->descriptorSetLayouts));
	this->pipelineLayouts.emplace("debugViews", std::make_unique<DebugViewsPipelineLayout>(window, &this->descriptorSetLayouts));
	this->pipelineLayouts.emplace("overVisualisation", std::make_unique<OverVisualisationPipelineLayout>(window, &this->descriptorSetLayouts));

	// Pipelines
	// (alot of the pipelines are identical other than their input pipeline layout, renderpass, and shader stages, possibly use the base class for these?)
	// (cubemap)shadow - shadow pass stage pipelines
	this->pipelines.emplace("shadow", std::make_unique<ShadowPipeline>(window, this->getPipelineLayout("shadow"), this->getRenderPass("shadow"), &this->sampleCountSetting, &this->shadowRes));
	this->pipelines.emplace("cubemapShadow", std::make_unique<CubemapShadowPipeline>(window, this->getPipelineLayout("cubemapShadow"), this->getRenderPass("shadow"), &this->sampleCountSetting, &this->shadowRes));
	// forward - regular forward shading pipeline
	this->pipelines.emplace("forward", std::make_unique<ForwardPipeline>(window, this->getPipelineLayout("forward"), this->getRenderPass("forward"), &this->sampleCountSetting, &this->shadowsEnabled));
	// deferredWriting - pipeline stage for writing to g-buffers
	this->pipelines.emplace("deferredWriting", std::make_unique<DeferredWritingPipeline>(window, this->getPipelineLayout("deferredWriting"), this->getRenderPass("deferred"), &this->sampleCountSetting));
	// deferredShading - pipeline stage for shading pass in deferred rendering
	this->pipelines.emplace("deferredShading", std::make_unique<DeferredShadingPipeline>(window, this->getPipelineLayout("deferredShading"), this->getRenderPass("deferred"), &this->sampleCountSetting, &this->shadowsEnabled));
	// post processing effects
	this->pipelines.emplace("mosaic", std::make_unique<MosaicPipeline>(window, this->getPipelineLayout("mosaic"), this->getRenderPass("postProcess")));
	this->pipelines.emplace("bloom", std::make_unique<BloomPipeline>(window, this->getPipelineLayout("bloom"), this->getRenderPass("postProcess")));
	this->pipelines.emplace("composition", std::make_unique<CompositionPipeline>(window, this->getPipelineLayout("composition"), this->getRenderPass("postProcess")));

	// Debug pipelines
	this->pipelines.emplace("lineDebug", std::make_unique<LineDebugPipeline>(window, this->getPipelineLayout("lineDebug"), this->getRenderPass("sunView"), &this->sampleCountSetting));
	this->pipelines.emplace("debugViews", std::make_unique<DebugViewsPipeline>(window, this->getPipelineLayout("debugViews"), this->getRenderPass("forward"), &this->sampleCountSetting));
	this->pipelines.emplace("overVisualisation", std::make_unique<OverVisualisationPipeline>(window, this->getPipelineLayout("overVisualisation"), this->getRenderPass("forward"), &this->sampleCountSetting));

	// Texture Buffers
	// colour - output buffer after geometry and lighting
	this->textureBuffers.emplace("colour", std::make_unique<ColourTextureBuffer>(&this->context));
	// brightness - buffer used to render scene brightness as input for bloom post processing
	this->textureBuffers.emplace("brightness", std::make_unique<ColourTextureBuffer>(&this->context));
	// intermediate - intermediate buffers used during post processing effects
	this->textureBuffers.emplace("intermediate", std::make_unique<ColourTextureBuffer>(&this->context));
	this->textureBuffers.emplace("intermediate2", std::make_unique<ColourTextureBuffer>(&this->context));
	// depth - standard depth buffer
	this->textureBuffers.emplace("depth", std::make_unique<DepthTextureBuffer>(&this->context));
	// gBuffers - g-buffers used in deferred rendering
	this->textureBuffers.emplace("gBuffer1", std::make_unique<ColourTextureBuffer>(&this->context, VK_FORMAT_A2R10G10B10_UNORM_PACK32));
	this->textureBuffers.emplace("gBuffer2", std::make_unique<ColourTextureBuffer>(&this->context, VK_FORMAT_R8G8B8A8_UNORM));
	this->textureBuffers.emplace("gBuffer3", std::make_unique<ColourTextureBuffer>(&this->context, VK_FORMAT_R8G8B8A8_UNORM));
	// blurOutput - output of blur in bloom post processing
	this->textureBuffers.emplace("blurOutput", std::make_unique<ColourTextureBuffer>(&this->context));

	// Debug texture buffers
	// sunView - buffer used to render the sun view
	this->textureBuffers.emplace("sunView", std::make_unique<ColourTextureBuffer>(&this->context));

	// Framebuffers
	// forward - 1 colour, 1 depth render targets
	this->framebuffers.emplace("forward", std::make_unique<ForwardFramebuffer>(window, &this->textureBuffers, this->getRenderPass("forward"), &this->sampleCountSetting));
	// deferred - 4 colour (3 g-buffers, 1 colour), 1 depth render targets
	this->framebuffers.emplace("deferred", std::make_unique<DeferredFramebuffer>(window, &this->textureBuffers, this->getRenderPass("deferred"), &this->sampleCountSetting));
	// sun - 1 colour (sunView buffer), 1 depth render targets
	this->framebuffers.emplace("sun", std::make_unique<SunFramebuffer>(window, &this->textureBuffers, this->getRenderPass("sunView"), &this->sampleCountSetting));
	// gui - writes directly to swapchain buffer
	this->framebuffers.emplace("gui", std::make_unique<GUIFramebuffer>(window, this->getRenderPass("gui")));
	// writeTo(output/intermediate) - 2 framebuffers to ping-pong writing to "colour" and "intermediate" buffers during post processing
	this->framebuffers.emplace("writeToOutput", std::make_unique<OutputFramebuffer>(window, &this->textureBuffers, this->getRenderPass("postProcess")));
	this->framebuffers.emplace("writeToIntermediate", std::make_unique<IntermediateFramebuffer>(window, &this->textureBuffers, this->getRenderPass("postProcess")));
	this->framebuffers.emplace("writeToIntermediate2", std::make_unique<Intermediate2Framebuffer>(window, &this->textureBuffers, this->getRenderPass("postProcess")));
	// writeToBlur - writes to blurOutput
	this->framebuffers.emplace("writeToBlur", std::make_unique<BlurFramebuffer>(window, &this->textureBuffers, this->getRenderPass("postProcess")));

	// Uniform Buffers
	VkPipelineStageFlags VFstageFlags = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	VkPipelineStageFlags VstageFlags = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT;
	VkPipelineStageFlags FstageFlags = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

	this->uniformBuffers.emplace("mvp", std::make_unique<UniformBuffer<glsl::MVPUniform>>(allocator, VFstageFlags, &this->uniforms.mvpUniform));
	this->uniformBuffers.emplace("cameraPlanes", std::make_unique<UniformBuffer<glsl::CameraPlanesUniform>>(allocator, FstageFlags, &this->uniforms.cameraPlanesUniform));

	// Synchronisation
	for (std::size_t i = 0; i < window->swapViews.size(); i++) {
		this->cmdBuffers.emplace_back(VkUtils::createCommandBuffer(*window, window->device->cmdPool));
		this->frameDoneFences.emplace_back(VkUtils::createFence(*window, VK_FENCE_CREATE_SIGNALED_BIT));
		this->imageAvailableSemaphores.emplace_back(VkUtils::createSemaphore(*window));
		this->renderFinishedSemaphores.emplace_back(VkUtils::createSemaphore(*window));
	}

	// Samplers
	SamplerInfo defaultSamplerInfo = {
		.magFilter = VK_FILTER_LINEAR,
		.minFilter = VK_FILTER_LINEAR,
		.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.anisotropyEnable = VK_TRUE };
	this->defaultSampler = VkUtils::createTextureSampler(*window, defaultSamplerInfo);
	SamplerInfo shadowMapSamplerInfo = {
		.magFilter = VK_FILTER_LINEAR,
		.minFilter = VK_FILTER_LINEAR,
		.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
		.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
		.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
		.compareEnable = 1, 
		.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL
	};
	this->shadowMapSampler = VkUtils::createTextureSampler(*window, shadowMapSamplerInfo);
	// Need a separate sampler that clamps to edge when sampling from textures such as the
	// brightness one for bloom ppe
	SamplerInfo brightnessSamplerInfo = {
		.magFilter = VK_FILTER_LINEAR,
		.minFilter = VK_FILTER_LINEAR,
		.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE 
	};
	this->brightnessSampler = VkUtils::createTextureSampler(*window, brightnessSamplerInfo);

	// Descriptor Sets
	std::vector<DescriptorBufferSetting> mvpDescriptorSettings = {
		{ this->getUniformBuffer("mvp")->getHandle(), VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER } };
	std::vector<DescriptorBufferSetting> cameraPlanesDescriptorSettings = {
		{ this->getUniformBuffer("cameraPlanes")->getHandle(), VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER } };
	std::vector<DescriptorImageSetting> deferredInputsDescriptorSettings = {
		{ this->getTextureBuffer("gBuffer1"), VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_NULL_HANDLE },
		{ this->getTextureBuffer("gBuffer2"), VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_NULL_HANDLE },
		{ this->getTextureBuffer("gBuffer3"), VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_NULL_HANDLE },
		{ this->getTextureBuffer("depth"), VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_NULL_HANDLE } };
	std::vector<DescriptorImageSetting> sunViewDescriptorSettings = {
		{ this->getTextureBuffer("sunView"), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, this->defaultSampler.handle } };
	std::vector<DescriptorImageSetting> colourOuputDescriptorSettings = {
		{ this->getTextureBuffer("colour"), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, this->defaultSampler.handle } };
	std::vector<DescriptorImageSetting> intermediateImageDescriptorSettings = {
		{ this->getTextureBuffer("intermediate"), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, this->defaultSampler.handle } };
	std::vector<DescriptorImageSetting> intermediate2ImageDescriptorSettings = {
		{ this->getTextureBuffer("intermediate2"), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, this->brightnessSampler.handle } };
	std::vector<DescriptorImageSetting> brightnessImageDescriptorSettings = {
		{ this->getTextureBuffer("brightness"), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, this->brightnessSampler.handle } };
	std::vector<DescriptorImageSetting> blurImageDescriptorSettings = {
		{ this->getTextureBuffer("blurOutput"), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, this->brightnessSampler.handle } };

	this->descriptorSets.emplace("mvp", std::make_unique<BufferDescriptorSet>(window, &this->descriptorSetLayouts.at("uboVF").handle, mvpDescriptorSettings));
	this->descriptorSets.emplace("cameraPlanes", std::make_unique<BufferDescriptorSet>(window, &this->descriptorSetLayouts.at("uboF").handle, cameraPlanesDescriptorSettings));
	this->descriptorSets.emplace("deferredInputs", std::make_unique<ImageDescriptorSet>(window, &this->descriptorSetLayouts.at("deferredInputAttachments").handle, deferredInputsDescriptorSettings));
	this->descriptorSets.emplace("sunView", std::make_unique<ImageDescriptorSet>(window, &this->descriptorSetLayouts.at("imageF").handle, sunViewDescriptorSettings));
	this->descriptorSets.emplace("sceneOutput", std::make_unique<ImageDescriptorSet>(window, &this->descriptorSetLayouts.at("imageF").handle, colourOuputDescriptorSettings));
	this->descriptorSets.emplace("intermediate", std::make_unique<ImageDescriptorSet>(window, &this->descriptorSetLayouts.at("imageF").handle, intermediateImageDescriptorSettings));
	this->descriptorSets.emplace("intermediate2", std::make_unique<ImageDescriptorSet>(window, &this->descriptorSetLayouts.at("imageF").handle, intermediate2ImageDescriptorSettings));
	this->descriptorSets.emplace("brightness", std::make_unique<ImageDescriptorSet>(window, &this->descriptorSetLayouts.at("imageF").handle, brightnessImageDescriptorSettings));
	this->descriptorSets.emplace("blurOutput", std::make_unique<ImageDescriptorSet>(window, &this->descriptorSetLayouts.at("imageF").handle, blurImageDescriptorSettings));

	// Post processing effects
	this->postProcessingEffects.emplace_back(std::make_pair("bloom", std::make_unique<BloomPostProcess>(this)));
	this->postProcessingEffects.emplace_back(std::make_pair("mosaic", std::make_unique<MosaicPostProcess>(this)));
}

Renderer::~Renderer() {
	// Used in case Vulkan/VMA throws an error during rendering,
	// we need to manually destroy ImGUI related Vulkan objects
	// through its shutdown methods before destroying our Vulkan 
	// device instance
	if (!this->handledImGUIShutdown) {
		vkDeviceWaitIdle(this->context.window->device->device);
		RendererUtils::destroyImGUI();
	}
}

void Renderer::setLights(std::vector<Light>* lights) {
	this->lights = lights;

	VulkanWindow* window = this->context.window.get();
	VulkanAllocator* allocator = this->context.allocator.get();

	this->numPointLights = 0;
	this->numDirectionalLights = 0;
	this->numSpotLights = 0;

	// Iterate over lights and find number of each lights
	for (std::size_t i = 0; i < lights->size(); i++) {
		Light light = lights->at(i);

		switch (light.getLightType()) {
		case LightType::POINT:
			this->numPointLights++;
			break;
		case LightType::DIRECTIONAL:
			this->numDirectionalLights++;
			break;
		case LightType::SPOT:
			this->numSpotLights++;
			break;
		}
	}

	// Create a texture buffers and framebuffers for array shadow maps for non-zero light types
	this->textureBuffers.emplace("pointArrayShadows", std::make_unique<CubemapArrayDepthTextureBuffer>(&this->context, this->numPointLights, VK_FORMAT_D32_SFLOAT, &this->shadowRes));
	this->textureBuffers.emplace("directionalShadow", std::make_unique<DepthTextureBuffer>(&this->context, VK_FORMAT_D32_SFLOAT, nullptr, &this->shadowRes));
#ifndef NDEBUG
	this->textureBuffers.emplace("pointArrayShadowsDebug", std::make_unique<ArrayColourTextureBuffer>(&this->context, this->numPointLights * 6, VK_FORMAT_R16G16B16A16_SFLOAT, &this->shadowRes));
	this->textureBuffers.emplace("directionalShadowDebug", std::make_unique<ColourTextureBuffer>(&this->context, VK_FORMAT_R16G16B16A16_SFLOAT, nullptr, &this->shadowRes));
#endif

	if (this->numPointLights != 0) {
		std::initializer_list<TextureBuffer*> pointShadowTextures = {
			this->getTextureBuffer("pointArrayShadows"),
#ifndef NDEBUG
			this->getTextureBuffer("pointArrayShadowsDebug")
#endif
		};

		this->framebuffers.emplace("pointArrayShadows", std::make_unique<ArrayFramebuffer>(window, pointShadowTextures, this->getRenderPass("shadow"), this->numPointLights * 6, &this->shadowRes));
	}

	if (this->numDirectionalLights != 0) {
		std::initializer_list<TextureBuffer*> directionalShadowTextures = {
			this->getTextureBuffer("directionalShadow"),
#ifndef NDEBUG
			this->getTextureBuffer("directionalShadowDebug")
#endif
		};

		this->framebuffers.emplace("directionalShadow", std::make_unique<ShadowFramebuffer>(window, directionalShadowTextures, this->getRenderPass("shadow"), &this->shadowRes));
	}

	// Light type counters (surely I can make a better system than this)
	int pointLightIndex = 0;
	int spotLightIndex = 0;

	// Populate ssbos
	for (std::size_t i = 0; i < lights->size(); i++) {
		Light light = lights->at(i);

		glsl::Light lightStruct = {
			.position = light.getPosition(),
			.direction = light.getDirection(),
			.colour = light.getColour(),
			.metadata = glm::ivec3(static_cast<int>(light.getLightType()), 0, light.getIntensity())
		};

		switch (light.getLightType()) {
		case LightType::POINT:
		{
			lightStruct.metadata.y = pointLightIndex;
			pointLightIndex++;
			break;
		}
		case LightType::DIRECTIONAL:
		{
			//LightMatrices lightMatrices = this->getLightMatricesForCameraFrustum(lightStruct);
			this->sunMatrices = this->getSunViewMatrices(lightStruct);
			this->sunLightIndex = i;
			this->ssbos.lightMatrices.emplace_back(this->sunMatrices.projection * this->sunMatrices.view);
			break;
		}
		case LightType::SPOT:
		{
			lightStruct.metadata.y = spotLightIndex;
			spotLightIndex++;
			break;
		}
		}

		this->ssbos.lights.emplace_back(lightStruct);
	}

	// Create SSBOs
	this->shaderStorageBuffers.emplace("lights", std::make_unique<ShaderStorageBuffer<glsl::Light>>(&this->context, &this->ssbos.lights));
	this->shaderStorageBuffers.emplace("lightMatrices", std::make_unique<ShaderStorageBuffer<glm::mat4>>(&this->context, &this->ssbos.lightMatrices));

	// Create descriptor sets
	std::vector<DescriptorImageSetting> pointLightShadowsDescriptorSettings = {
		{ this->getTextureBuffer("pointArrayShadows"), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, this->shadowMapSampler.handle } };
	std::vector<DescriptorImageSetting> directionalLightShadowDescriptorSettings = {
		{ this->getTextureBuffer("directionalShadow"), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, this->shadowMapSampler.handle } };

	this->descriptorSets.emplace("pointLightShadows", std::make_unique<ImageDescriptorSet>(window, &this->descriptorSetLayouts.at("imageF").handle, pointLightShadowsDescriptorSettings));
	this->descriptorSets.emplace("directionalLightShadow", std::make_unique<ImageDescriptorSet>(window, &this->descriptorSetLayouts.at("imageF").handle, directionalLightShadowDescriptorSettings));

#ifndef NDEBUG
	std::vector<DescriptorImageSetting> pointLightShadowsDebugDescriptorSettings = {
		{ this->getTextureBuffer("pointArrayShadowsDebug"), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, this->shadowMapSampler.handle } };
	std::vector<DescriptorImageSetting> directionalLightShadowDebugDescriptorSettings = {
		{ this->getTextureBuffer("directionalShadowDebug"), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, this->shadowMapSampler.handle } };

	this->descriptorSets.emplace("pointLightShadowsDebug", std::make_unique<ArrayImageDescriptorSet>(window, &this->descriptorSetLayouts.at("imageF").handle, pointLightShadowsDebugDescriptorSettings));
	this->descriptorSets.emplace("directionalLightShadowDebug", std::make_unique<ImageDescriptorSet>(window, &this->descriptorSetLayouts.at("imageF").handle, directionalLightShadowDebugDescriptorSettings));
#endif

	std::vector<DescriptorBufferSetting> lightSSBODescriptorSettings = {
		{ this->getShaderStorageBuffer("lights")->getHandle(), VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, this->getShaderStorageBuffer("lights")->getBufferSize() } };

	std::vector<DescriptorBufferSetting> lightMatricesSSBODescriptorSettings = {
		{ this->getShaderStorageBuffer("lightMatrices")->getHandle(), VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, this->getShaderStorageBuffer("lightMatrices")->getBufferSize() } };

	this->descriptorSets.emplace("lights", std::make_unique<BufferDescriptorSet>(window, &this->descriptorSetLayouts.at("ssboF").handle, lightSSBODescriptorSettings));
	this->descriptorSets.emplace("lightMatrices", std::make_unique<BufferDescriptorSet>(window, &this->descriptorSetLayouts.at("ssboF").handle, lightMatricesSSBODescriptorSettings));
}

bool Renderer::checkSwapchain() {
	if (!this->recreateSwapchain)
		return false;

	// Handle minimisation
	GLFWwindow* window = this->context.window->window;

	int width, height;
	glfwGetFramebufferSize(window, &width, &height);
	// Loop indefinitely until framebuffer size becomes non-zero (i.e. no longer minimised)
	while (width == 0 || height == 0) {
		glfwGetFramebufferSize(window, &width, &height);
		glfwWaitEvents();
	}

	// Wait for GPU to finish processing
	vkDeviceWaitIdle(this->context.window->device->device);

	const SwapChanges swapChanges = ::recreateSwapchain(*this->context.window);

	if (swapChanges.changedFormat || this->forceRecreate)
		this->recreateFormatDependents();
	if (swapChanges.changedSize || this->forceRecreate)
		this->recreateSizeDependents();

	// Always recreate swapchain image view dependents
	this->recreateSwapViewDependents();

	this->recreateSwapchain = false;
	this->forceRecreate = false;

	return true;
}

bool Renderer::acquireSwapchainImage() {
	this->frameIndex++;
	this->frameIndex %= this->cmdBuffers.size();

	VkUtils::waitForFences(*this->context.window, this->frameDoneFences, this->frameIndex);

	if (const VkResult res = VkUtils::acquireNextSwapchainImage(*this->context.window, this->imageAvailableSemaphores, this->frameIndex, this->imageIndex);
		res == VK_SUBOPTIMAL_KHR || res == VK_ERROR_OUT_OF_DATE_KHR) {
		this->recreateSwapchain = true;

		// If vkAcquireNextImageKHR returned VK_SUBOPTIMAL_KHR we can still render the frame and
		// recreate the swapchain before the next frame, this way the signalled semaphore from
		// vkAcquireNextImageKHR will be waited on during this frame's submission.
		if (res == VK_SUBOPTIMAL_KHR) {
			VkUtils::resetFences(*this->context.window, this->frameDoneFences, this->frameIndex);

			return false;
		}
		// Else if vkAcquireNextImageKHR returned VK_ERROR_OUT_OF_DATE_KHR we need to immediately
		// recreate the swapchain without trying to render the current frame, this is fine since
		// if vkAcquireNextImageKHR returns VK_ERROR_OUT_OF_DATE_KHR the imageAvailableSemaphore
		// is not signalled

		--this->frameIndex;
		this->frameIndex %= this->cmdBuffers.size();

		return true;
	}

	VkUtils::resetFences(*this->context.window, this->frameDoneFences, this->frameIndex);

	return false;
}

void Renderer::update(float timeDelta) {
	this->camera.update(this->context.window->window, timeDelta);

	this->uniforms.mvpUniform.projection = this->camera.getProjectionMat();
	this->uniforms.mvpUniform.view = this->camera.getViewMat();
	this->uniforms.mvpUniform.camPos = glm::vec4(this->camera.getPosition(), 1.0f);
	this->uniforms.cameraPlanesUniform._far = this->camera.getFarPlane();
	this->uniforms.cameraPlanesUniform._near = this->camera.getNearPlane();

	// Update any light data
	int directionalLightIndex = 0;

	for (std::size_t i = 0; i < this->lights->size(); i++) {
		Light light = lights->at(i);

		glsl::Light lightStruct = this->ssbos.lights.at(i);

		switch (light.getLightType()) {
		case LightType::POINT:
		{
			break;
		}
		case LightType::DIRECTIONAL:
		{
			// Sun light needs cascaded shadow mapping since it encompasses the entire
			// camera frustum, and so the area covered by a single pixel of the shadow map is
			// large and results in pixelated shadows close to the camera
			//LightMatrices lightMatrices = this->getLightMatricesForCameraFrustum(lightStruct);

			this->sunMatrices = this->getSunViewMatrices(lightStruct);

			// Update light matrix
			this->ssbos.lightMatrices.at(directionalLightIndex) = this->sunMatrices.projection * this->sunMatrices.view;

			directionalLightIndex++;
			break;
		}
		case LightType::SPOT:
		{
			break;
		}
		}
	}

	// TODO: Move all this somewhere else, and only calculate it if the debug
	// option for frustum bounds is actually enabled.
	// Update debug frustum lines
	std::array<glm::vec4, 8> frustumCornersArr = this->camera.getFrustumCorners();
	std::vector<glm::vec4> frustumCorners(frustumCornersArr.begin(), frustumCornersArr.end());
	std::vector<glm::vec3> lineColours(8, glm::vec3(1.0f));
	std::vector<std::uint32_t> lineIndices = {
		0, 1, 1, 2, 2, 3, 3, 0,
		0, 4, 1, 5, 2, 6, 3, 7,
		4, 5, 5, 6, 6, 7, 7, 4
	};

	// GPU buffers
	if (!this->lineMeshDataInit) {
		vk::Buffer posLineGPU = vk::createBuffer(
			*this->context.allocator,
			8 * sizeof(glm::vec4),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			0,
			VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

		vk::Buffer colLineGPU = vk::createBuffer(
			*this->context.allocator,
			8 * sizeof(glm::vec3),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			0,
			VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

		vk::Buffer indexLineGPU = vk::createBuffer(
			*this->context.allocator,
			24 * sizeof(std::uint32_t),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			0,
			VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

		// Staging buffers
		vk::Buffer posStaging = vk::createBuffer(
			*this->context.allocator,
			8 * sizeof(glm::vec4),
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

		vk::Buffer colStaging = vk::createBuffer(
			*this->context.allocator,
			8 * sizeof(glm::vec3),
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

		vk::Buffer indexStaging = vk::createBuffer(
			*this->context.allocator,
			24 * sizeof(std::uint32_t),
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

		BakedModelLoader::mapToGPU(*this->context.allocator, posLineGPU, posStaging, frustumCorners);
		BakedModelLoader::mapToGPU(*this->context.allocator, colLineGPU, colStaging, lineColours);
		BakedModelLoader::mapToGPU(*this->context.allocator, indexLineGPU, indexStaging, lineIndices);

		VkCommandBuffer uploadCmd = VkUtils::createCommandBuffer(*this->context.window, this->context.window->device->cmdPool);

		VkUtils::beginCommandBuffer(uploadCmd);

		BakedModelLoader::copyToGPU(uploadCmd, posLineGPU, posStaging, frustumCorners);
		BakedModelLoader::copyToGPU(uploadCmd, colLineGPU, colStaging, lineColours);
		BakedModelLoader::copyToGPU(uploadCmd, indexLineGPU, indexStaging, lineIndices);

		VkUtils::endAndSubmitCommandBuffer(*this->context.window, uploadCmd);

		this->lineMeshData = LineMeshData{
			std::move(posLineGPU),
			std::move(colLineGPU),
			std::move(indexLineGPU),
			std::move(posStaging),
			std::move(colStaging),
			std::move(indexStaging),
			lineIndices.size()
		};

		this->lineMeshDataInit = true;
	} else {
		BakedModelLoader::mapToGPU(*this->context.allocator, this->lineMeshData.posBuffer, this->lineMeshData.posBufferStaging, frustumCorners);
		BakedModelLoader::mapToGPU(*this->context.allocator, this->lineMeshData.colBuffer, this->lineMeshData.colBufferStaging, lineColours);
		BakedModelLoader::mapToGPU(*this->context.allocator, this->lineMeshData.indicesBuffer, this->lineMeshData.indicesBufferStaging, lineIndices);

		VkCommandBuffer uploadCmd = VkUtils::createCommandBuffer(*this->context.window, this->context.window->device->cmdPool);

		VkUtils::beginCommandBuffer(uploadCmd);

		BakedModelLoader::copyToGPU(uploadCmd, this->lineMeshData.posBuffer, this->lineMeshData.posBufferStaging, frustumCorners);
		BakedModelLoader::copyToGPU(uploadCmd, this->lineMeshData.colBuffer, this->lineMeshData.colBufferStaging, lineColours);
		BakedModelLoader::copyToGPU(uploadCmd, this->lineMeshData.indicesBuffer, this->lineMeshData.indicesBufferStaging, lineIndices);

		VkUtils::endAndSubmitCommandBuffer(*this->context.window, uploadCmd);
	}
}

void Renderer::render() {
	// Begin command buffer
	RendererUtils::bindCommandBuffer(this->cmdBuffers[this->frameIndex]);
	RendererUtils::beginCommandBuffer(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
	this->driver->getTimestampManager().resetGPUQueryPool();
	this->driver->getTimestampManager().writeGPUTimestamp("entireFrame", VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);

	// Update uniform and shader storage buffers
	RendererUtils::updateUniformBuffer(this->getUniformBuffer("mvp"));
	RendererUtils::updateShaderStorageBuffer(this->getShaderStorageBuffer("lights"));
	RendererUtils::updateShaderStorageBuffer(this->getShaderStorageBuffer("lightMatrices"));

	// Debug pass (if its enabled)
	if (this->debugView) {
		this->renderDebugViews();
		return;
	}

	// Shadow pass
	if (this->shadowsEnabled) {
		RendererUtils::updateUniformBuffer(this->getUniformBuffer("cameraPlanes"));
		this->renderShadowMaps();
	}

	// Transition any dummy light textures to respective layout
	if (this->numPointLights == 0) {
		RendererUtils::imageBarrier(this->getTextureBuffer("pointArrayShadows")->getImage().image, 0, 0,
			VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
			VkImageSubresourceRange{ VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 6 });
	}
	if (this->numDirectionalLights == 0) {
		RendererUtils::imageBarrier(this->getTextureBuffer("directionalShadow")->getImage().image, 0, 0,
			VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
			VkImageSubresourceRange{ VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 });
	}

	// Scene pass
	if (this->renderingType == 1) {
		this->renderDeferred();
	} else {
		this->renderForward();
	}

	// Post processing effects
	VkDescriptorSet readImage = this->getDescriptorSet("sceneOutput")->getHandle();
	VkDescriptorSet writeImage = this->getDescriptorSet("intermediate")->getHandle();

	Framebuffer* framebuffer1 = this->getFramebuffer("writeToIntermediate");
	Framebuffer* framebuffer2 = this->getFramebuffer("writeToOutput");

	bool useIntermediate = false;

	for (const auto& [effectName, effect] : this->postProcessingEffects) {
		if (!effect->getEnabled()) continue;
		useIntermediate = !useIntermediate;

		effect->apply(framebuffer1, this->imageIndex, readImage);

		std::swap(readImage, writeImage);
		std::swap(framebuffer1, framebuffer2);
	}

	// Blit image to swapchain
	VkImage srcImage = useIntermediate ?
		this->getTextureBuffer("intermediate")->getImage().image :
		this->getTextureBuffer("colour")->getImage().image;
	VkImage swapchainImage = this->context.window->swapImages[this->imageIndex];

	RendererUtils::blitImageToSwapchain(
		srcImage, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, swapchainImage,
		this->context.window->swapchainExtent, VK_FILTER_LINEAR);

	// Render GUI
	this->driver->getTimestampManager().writeGPUTimestamp("gui", VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
	RendererUtils::beginRenderPass(this->getRenderPass("gui"), this->getFramebuffer("gui"), this->imageIndex);
	RendererUtils::renderImGUI();
	RendererUtils::endRenderPass();
	this->driver->getTimestampManager().writeGPUTimestamp("gui", VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

	this->driver->getTimestampManager().writeGPUTimestamp("entireFrame", VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
	RendererUtils::endCommandBuffer();
}

void Renderer::renderForward() {
	std::vector<MeshData>& meshData = this->driver->getMeshData();

	this->driver->getTimestampManager().writeGPUTimestamp("forward", VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);

	RendererUtils::beginRenderPass(this->getRenderPass("forward"), this->getFramebuffer("forward"), this->imageIndex);
	RendererUtils::bindGraphicPipeline(this->getPipeline("forward")->getHandle());

	glsl::LightsAndEmissive lightsAndEmissive = {
		.numLights = this->numLights,
		.emissiveStrength = this->emissiveStrength,
		.brightnessThreshold = this->brightnessThreshold,
		.shadowBias = this->shadowBias
	};

	RendererUtils::bindPushConstant(
		this->getPipelineLayout("forward")->getHandle(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(glsl::LightsAndEmissive), &lightsAndEmissive);

	std::vector<VkDescriptorSet> descriptorSets;
	// Add descriptors that will always be present in same order as in shader
	descriptorSets.emplace_back(this->getDescriptorSet("mvp")->getHandle());
	descriptorSets.emplace_back(this->getDescriptorSet("lights")->getHandle());

	if (this->shadowsEnabled) {
		descriptorSets.emplace_back(this->getDescriptorSet("pointLightShadows")->getHandle());
		descriptorSets.emplace_back(this->getDescriptorSet("directionalLightShadow")->getHandle());
		descriptorSets.emplace_back(this->getDescriptorSet("cameraPlanes")->getHandle());
		descriptorSets.emplace_back(this->getDescriptorSet("lightMatrices")->getHandle());
	}

	RendererUtils::bindGraphicDescriptorSets(
		this->getPipelineLayout("forward")->getHandle(), 0, descriptorSets.size(),
		descriptorSets.data());

	RendererUtils::setCullMode(VK_CULL_MODE_BACK_BIT);

	auto perMeshCallback = [this, &descriptorSets](MeshData& meshData) {
		// Material descriptor should always be last descriptor since its per mesh
		RendererUtils::bindGraphicDescriptorSets(
			this->getPipelineLayout("forward")->getHandle(), descriptorSets.size(), 1,
			&this->driver->getMaterialDescriptors().at(meshData.materialId));
		};

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

	this->driver->getTimestampManager().writeGPUTimestamp("forward", VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

	if (this->sunLightIndex == -1 || (!this->showSunView)) return;

	// Update MVP uniform for sun debug view
	glsl::Light lightStruct = this->ssbos.lights.at(this->sunLightIndex);
	this->uniforms.mvpUniform.projection = this->sunMatrices.projection;
	this->uniforms.mvpUniform.view = this->sunMatrices.view;
	this->uniforms.mvpUniform.camPos = glm::vec4(lightStruct.position, 1.0f);

	RendererUtils::updateUniformBuffer(this->getUniformBuffer("mvp"));

	// Sun position view
	RendererUtils::beginRenderPass(this->getRenderPass("sunView"), this->getFramebuffer("sun"), this->imageIndex);
	RendererUtils::bindGraphicPipeline(this->getPipeline("forward")->getHandle());

	RendererUtils::bindPushConstant(
		this->getPipelineLayout("forward")->getHandle(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(glsl::LightsAndEmissive), &lightsAndEmissive);

	RendererUtils::bindGraphicDescriptorSets(
		this->getPipelineLayout("forward")->getHandle(), 0, descriptorSets.size(),
		descriptorSets.data());

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

	// Render frustum bounding box
	if (this->renderCameraFrustumBounds) {
		RendererUtils::bindGraphicPipeline(this->getPipeline("lineDebug")->getHandle());

		RendererUtils::bindGraphicDescriptorSets(
			this->getPipelineLayout("lineDebug")->getHandle(), 0, 1,
			&this->getDescriptorSet("mvp")->getHandle());

		RendererUtils::drawLineMesh(this->lineMeshData);
	}

	RendererUtils::endRenderPass();
}

void Renderer::renderDeferred() {
	std::vector<MeshData>& meshData = this->driver->getMeshData();

	this->driver->getTimestampManager().writeGPUTimestamp("deferred", VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);

	// Writing to G-buffers pass
	RendererUtils::beginRenderPass(this->getRenderPass("deferred"), this->getFramebuffer("deferred"), this->imageIndex);
	RendererUtils::bindGraphicPipeline(this->getPipeline("deferredWriting")->getHandle());

	RendererUtils::bindGraphicDescriptorSets(
		this->getPipelineLayout("deferredWriting")->getHandle(), 0, 1,
		&this->getDescriptorSet("mvp")->getHandle());

	auto perMeshCallback = [this](MeshData& meshData) {
		RendererUtils::bindGraphicDescriptorSets(
			this->getPipelineLayout("deferredWriting")->getHandle(), 1, 1,
			&this->driver->getMaterialDescriptors().at(meshData.materialId));
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

	// Shading pass
	RendererUtils::nextSubpass();
	RendererUtils::bindGraphicPipeline(this->getPipeline("deferredShading")->getHandle());

	glsl::LightsAndEmissive lightsAndEmissive = {
		.numLights = this->numLights,
		.emissiveStrength = this->emissiveStrength,
		.brightnessThreshold = this->brightnessThreshold,
		.shadowBias = this->shadowBias
	};

	RendererUtils::bindPushConstant(
		this->getPipelineLayout("deferredShading")->getHandle(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(glsl::LightsAndEmissive), &lightsAndEmissive);

	std::vector<VkDescriptorSet> descriptorSets;
	descriptorSets.emplace_back(this->getDescriptorSet("deferredInputs")->getHandle());
	descriptorSets.emplace_back(this->getDescriptorSet("mvp")->getHandle());
	descriptorSets.emplace_back(this->getDescriptorSet("lights")->getHandle());

	if (this->shadowsEnabled) {
		descriptorSets.emplace_back(this->getDescriptorSet("pointLightShadows")->getHandle());
		descriptorSets.emplace_back(this->getDescriptorSet("directionalLightShadow")->getHandle());
		descriptorSets.emplace_back(this->getDescriptorSet("cameraPlanes")->getHandle());
		descriptorSets.emplace_back(this->getDescriptorSet("lightMatrices")->getHandle());
	}

	RendererUtils::bindGraphicDescriptorSets(
		this->getPipelineLayout("deferredShading")->getHandle(), 0, descriptorSets.size(),
		descriptorSets.data());

	RendererUtils::drawDirect(3, 1, 0, 0);

	RendererUtils::endRenderPass();

	this->driver->getTimestampManager().writeGPUTimestamp("deferred", VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
}

void Renderer::renderDebugViews() {
	std::vector<MeshData>& meshData = this->driver->getMeshData();

	this->driver->getTimestampManager().writeGPUTimestamp("debugViews", VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);

	// Determine debug state flags
	bool isOvervisualisation = this->debugState > 6;

	if (isOvervisualisation) {
		// Set clear colour to a dark green to create a 'negative'-like image, but also restore
		// the original clear colour afterwards for when we disable overvisualisation
		VkClearValue originalValue = this->getRenderPass("forward")->getClearValues().at(0);
		this->getRenderPass("forward")->getClearValues().at(0) = { {0.0f, 0.3f, 0.0f, 1.0f} };
		RendererUtils::beginRenderPass(this->getRenderPass("forward"), this->getFramebuffer("forward"), this->imageIndex);
		this->getRenderPass("forward")->getClearValues().at(0) = originalValue;
	} else {
		RendererUtils::beginRenderPass(this->getRenderPass("forward"), this->getFramebuffer("forward"), this->imageIndex);
	}

	std::vector<VkDescriptorSet> descriptorSets;
	descriptorSets.emplace_back(this->getDescriptorSet("mvp")->getHandle());

	if (!isOvervisualisation) {
		RendererUtils::bindGraphicPipeline(this->getPipeline("debugViews")->getHandle());

		debugStatePC debugState = {
			.lightCount = this->numLights,
			.debugState = this->debugState
		};

		RendererUtils::bindPushConstant(
			this->getPipelineLayout("debugViews")->getHandle(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(debugStatePC), &debugState);

		descriptorSets.emplace_back(this->getDescriptorSet("lights")->getHandle());
		descriptorSets.emplace_back(this->getDescriptorSet("cameraPlanes")->getHandle());

		RendererUtils::bindGraphicDescriptorSets(
			this->getPipelineLayout("debugViews")->getHandle(), 0, descriptorSets.size(),
			descriptorSets.data());

		RendererUtils::setCullMode(VK_CULL_MODE_BACK_BIT);
	} else {
		RendererUtils::bindGraphicPipeline(this->getPipeline("overVisualisation")->getHandle());

		RendererUtils::bindGraphicDescriptorSets(
			this->getPipelineLayout("overVisualisation")->getHandle(), 0, descriptorSets.size(),
			descriptorSets.data());
	}

	auto perMeshCallbackDebug = [this, &descriptorSets](MeshData& meshData) {
		RendererUtils::bindGraphicDescriptorSets(
			this->getPipelineLayout("debugViews")->getHandle(), descriptorSets.size(), 1,
			&this->driver->getMaterialDescriptors().at(meshData.materialId));
		};

	for (std::size_t i = 0; i < meshData.size(); i++) {
		if (!isOvervisualisation) {
			RendererUtils::drawMesh(meshData[i], perMeshCallbackDebug);
		} else {
			// Overdraw
			if (this->debugState == 7) {
				RendererUtils::setDepthTestEnable(VK_FALSE);
			}
			// Overshading
			else if (this->debugState == 8) {
				RendererUtils::setDepthTestEnable(VK_TRUE);
			}

			RendererUtils::drawMesh(meshData[i]);
		}
	}

	RendererUtils::endRenderPass();

	this->driver->getTimestampManager().writeGPUTimestamp("debugViews", VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

	// Blit image to swapchain
	VkImage srcImage = this->textureBuffers.at("colour")->getImage().image;
	VkImage swapchainImage = this->context.window->swapImages[this->imageIndex];

	RendererUtils::blitImageToSwapchain(
		srcImage, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, swapchainImage,
		this->context.window->swapchainExtent, VK_FILTER_LINEAR);

	// Render GUI
	RendererUtils::beginRenderPass(this->getRenderPass("gui"), this->getFramebuffer("gui"), this->imageIndex);
	RendererUtils::renderImGUI();
	RendererUtils::endRenderPass();

	this->driver->getTimestampManager().writeGPUTimestamp("entireFrame", VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
	RendererUtils::endCommandBuffer();
}

void Renderer::renderShadowMaps() {
	std::vector<MeshData>& meshData = this->driver->getMeshData();

	this->driver->getTimestampManager().writeGPUTimestamp("shadows", VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);

	// Some light-specific counters
	std::uint32_t pointLightIndex = 0;
	std::uint32_t directionalLightIndex = 0;

	// For each light
	for (std::size_t i = 0; i < this->lights->size(); i++) {
		Light light = this->lights->at(i);

		switch (light.getLightType()) {
		case LightType::POINT:
		{
			assert(this->numPointLights != 0 && "Trying to render a point light shadow map but numPointLights is 0?");

			constexpr glm::vec3 directions[6] = {
				glm::vec3(1.0f, 0.0f, 0.0f),
				glm::vec3(-1.0f, 0.0f, 0.0f),
				glm::vec3(0.0f, 1.0f, 0.0f),
				glm::vec3(0.0f, -1.0f, 0.0f),
				glm::vec3(0.0f, 0.0f, 1.0f),
				glm::vec3(0.0f, 0.0f, -1.0f),
			};

			constexpr glm::vec3 upVectors[6] = {
				glm::vec3(0.0f, -1.0f, 0.0f),
				glm::vec3(0.0f, -1.0f, 0.0f),
				glm::vec3(0.0f, 0.0f, 1.0f),
				glm::vec3(0.0f, 0.0f, -1.0f),
				glm::vec3(0.0f, -1.0f, 0.0f),
				glm::vec3(0.0f, -1.0f, 0.0f),
			};

			const glm::mat4 cubePerspective = glm::perspective(glm::radians(90.0f), 1.0f, this->camera.getNearPlane(), this->camera.getFarPlane());

			// Render to each face of the cube map
			for (std::size_t face = 0; face < 6; face++) {
				// Calculate layer index
				std::uint32_t layer = (pointLightIndex * 6) + face;

				RendererUtils::beginRenderPass(this->getRenderPass("shadow"), this->getFramebuffer("pointArrayShadows"), layer);

				RendererUtils::bindGraphicPipeline(this->getPipeline("cubemapShadow")->getHandle());

				glm::mat4 cubeView = glm::lookAt(light.getPosition(), light.getPosition() + directions[face], upVectors[face]);
				glm::mat4 cubeMatrix = cubePerspective * cubeView;

				struct cubemapFragmentPC {
					glm::vec4 lightPos;
					float farPlane;
				};

				cubemapFragmentPC fragPC = {
					.lightPos = glm::vec4(light.getPosition(), 1.0f),
					.farPlane = this->camera.getFarPlane()
				};

				RendererUtils::bindPushConstant(this->getPipelineLayout("cubemapShadow")->getHandle(),
					VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), &cubeMatrix);
				RendererUtils::bindPushConstant(this->getPipelineLayout("cubemapShadow")->getHandle(),
					VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(glm::mat4), sizeof(cubemapFragmentPC), &fragPC);

				RendererUtils::setDepthBias(this->depthBiasConstant, 0.0f, this->depthBiasSlopeFactor);

				for (std::size_t i = 0; i < meshData.size(); i++)
					RendererUtils::drawMeshGeometry(meshData[i]);

				RendererUtils::endRenderPass();
			}

			pointLightIndex++;
			break;
		}
		case LightType::DIRECTIONAL:
		{
			assert(this->numDirectionalLights != 0 && "Trying to render a directional light shadow map but numDirectionalLights is 0?");

			RendererUtils::beginRenderPass(this->getRenderPass("shadow"), this->getFramebuffer("directionalShadow"), directionalLightIndex);

			RendererUtils::bindGraphicPipeline(this->getPipeline("shadow")->getHandle());

			glm::mat4 lightMatrix = this->ssbos.lightMatrices.at(directionalLightIndex);

			RendererUtils::bindPushConstant(this->getPipelineLayout("shadow")->getHandle(),
				VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), &lightMatrix);
#if !defined(NDEBUG)
			int projType = 1;

			RendererUtils::bindPushConstant(this->getPipelineLayout("shadow")->getHandle(),
				VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(glm::mat4), sizeof(int), &projType);

			RendererUtils::bindGraphicDescriptorSets(
				this->getPipelineLayout("shadow")->getHandle(), 0, 1,
				&this->getDescriptorSet("cameraPlanes")->getHandle());
#endif
			RendererUtils::setDepthBias(this->depthBiasConstant, 0.0f, this->depthBiasSlopeFactor);

			for (std::size_t i = 0; i < meshData.size(); i++)
				RendererUtils::drawMeshGeometry(meshData[i]);

			RendererUtils::endRenderPass();

			directionalLightIndex++;
			break;
		}
		case LightType::SPOT:
		{
			assert(this->numSpotLights != 0 && "Trying to render a spot light shadow map but numSpotLights is 0?");

			break;
		}
		}
	}

	this->driver->getTimestampManager().writeGPUTimestamp("shadows", VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
}

LightMatrices Renderer::getLightMatricesForCameraFrustum(glsl::Light& lightStruct) {
	std::array<glm::vec4, 8> frustumCorners = this->camera.getFrustumCorners();
	
	// Get frustum center
	glm::vec3 frustumCenter(0.0f);
	for (const glm::vec3& corner : frustumCorners)
		frustumCenter += corner;
	frustumCenter /= static_cast<float>(frustumCorners.size());

	// Calc light pos
	lightStruct.position = frustumCenter - lightStruct.direction * 50.0f;

	// Construct light view matrix
	glm::mat4 lightView = glm::lookAt(lightStruct.position, frustumCenter, glm::vec3(0.0f, 1.0f, 0.0f));

	// Get AABB of the transformed frustum
	glm::vec3 min(FLT_MAX);
	glm::vec3 max(-FLT_MAX);
	for (const glm::vec4& corner : frustumCorners) {
		glm::vec3 transformedCorner = lightView * corner;
		min = glm::min(min, transformedCorner);
		max = glm::max(max, transformedCorner);
	}

	// Add padding to depth
	if (min.z < 0)
		min.z *= zMult;
	else
		min.z /= zMult;

	if (max.z < 0)
		max.z /= zMult;
	else
		max.z *= zMult;

	// Construct light projection matrix
	glm::mat4 lightOrtho = glm::ortho(min.x, max.x, max.y, min.y, min.z, max.z);

	return { lightOrtho, lightView };
}

LightMatrices Renderer::getSunViewMatrices(glsl::Light& lightStruct) {
	glm::mat4 sunOrtho = glm::ortho(-this->sunOrthoBounds, this->sunOrthoBounds, this->sunOrthoBounds, -this->sunOrthoBounds, this->sunShadowNear, this->sunShadowFar);
	glm::mat4 sunView = glm::lookAt(-lightStruct.direction * this->sunDistance, glm::vec3(0.0f, 0.0f, -20.0f), glm::vec3(0.0f, 1.0f, 0.0f));

	return { sunOrtho, sunView };
}

void Renderer::submitRender() {
	const VkResult result = submitAndPresent(
		*this->context.window,
		this->cmdBuffers,
		this->frameDoneFences,
		this->imageAvailableSemaphores,
		this->renderFinishedSemaphores,
		this->frameIndex,
		this->imageIndex);

	if (result == VK_SUBOPTIMAL_KHR || result == VK_ERROR_OUT_OF_DATE_KHR)
		this->recreateSwapchain = true;
	else if (result != VK_SUCCESS)
		throw Utils::Error("Unable to present swapchain image %u\n vkQueuePresentKHR() returned %s", this->imageIndex, Utils::toString(result).c_str());

	// Read back timestamps from last frame to display in GUI
	this->driver->getTimestampManager().readBackGPUTimestamps();
}

void Renderer::finishRendering() {
	vkDeviceWaitIdle(this->context.window->device->device);
	RendererUtils::destroyImGUI();
	this->handledImGUIShutdown = true;
}

void Renderer::createDummyTexture() {
	vk::Image dummyImage = vk::createDummyImage(this->context, VK_FORMAT_R8G8B8A8_SRGB);
	vk::ImageView dummyImageView = vk::createImageView(this->context, dummyImage.image, VK_FORMAT_R8G8B8A8_SRGB);
	this->dummyTexture = { std::move(dummyImage), std::move(dummyImageView) };
}

void Renderer::recreateFormatDependents() {
	// Recreate render passes
	for (auto& renderPass : this->renderPasses)
		renderPass.second->recreate();
}

void Renderer::recreateSizeDependents() {
	// Recreate texture buffers
	// (causes dependent DescriptorSets to also be recreated)
	for (auto& textureBuffer : this->textureBuffers)
		textureBuffer.second->recreate();

	// Recreate pipeline layouts
	for (auto& pipelineLayout : this->pipelineLayouts)
		pipelineLayout.second->recreate();

	// Recreate pipelines
	for (auto& pipeline : this->pipelines)
		pipeline.second->recreate();
}

void Renderer::recreateSwapViewDependents() {
	// Recreate framebuffers
	for (auto& framebuffer : this->framebuffers)
		framebuffer.second->recreate();
}

Driver* Renderer::getDriver() {
	return this->driver;
}

VulkanContext& Renderer::getContext() {
	return this->context;
}

Camera& Renderer::getCamera() {
	return this->camera;
}

RenderPass* Renderer::getRenderPass(const std::string& renderPass) {
	RenderPass* ret = nullptr;

	try {
		ret = this->renderPasses.at(renderPass).get();
	} catch (const std::out_of_range&) {
		throw Utils::Error("Could not find: %s in 'renderPasses'\n", renderPass.c_str());
	}

	return ret;
}

VkDescriptorSetLayout Renderer::getDescriptorSetLayout(const std::string& descriptorSetLayout) {
	VkDescriptorSetLayout ret = VK_NULL_HANDLE;

	try {
		ret = this->descriptorSetLayouts.at(descriptorSetLayout).handle;
	} catch (const std::out_of_range&) {
		throw Utils::Error("Could not find: %s in 'descriptorSetLayouts'\n", descriptorSetLayout.c_str());
	}

	return ret;
}

PipelineLayout* Renderer::getPipelineLayout(const std::string& pipelineLayout) {
	PipelineLayout* ret = nullptr;

	try {
		ret = this->pipelineLayouts.at(pipelineLayout).get();
	} catch (const std::out_of_range&) {
		throw Utils::Error("Could not find: %s in 'pipelineLayouts'\n", pipelineLayout.c_str());
	}

	return ret;
}

Pipeline* Renderer::getPipeline(const std::string& pipeline) {
	Pipeline* ret = nullptr;

	try {
		ret = this->pipelines.at(pipeline).get();
	} catch (const std::out_of_range&) {
		throw Utils::Error("Could not find: %s in 'pipelines'\n", pipeline.c_str());
	}

	return ret;
}

Framebuffer* Renderer::getFramebuffer(const std::string& framebuffer) {
	Framebuffer* ret = nullptr;

	try {
		ret = this->framebuffers.at(framebuffer).get();
	} catch (const std::out_of_range&) {
		throw Utils::Error("Could not find: %s in 'framebuffer'\n", framebuffer.c_str());
	}

	return ret;
}

TextureBuffer* Renderer::getTextureBuffer(const std::string& textureBuffer) {
	TextureBuffer* ret = nullptr;

	try {
		ret = this->textureBuffers.at(textureBuffer).get();
	} catch (const std::out_of_range&) {
		throw Utils::Error("Could not find: %s in 'textureBuffers'\n", textureBuffer.c_str());
	}

	return ret;
}

IUniformBuffer* Renderer::getUniformBuffer(const std::string& uniformBuffer) {
	IUniformBuffer* ret = nullptr;

	try {
		ret = this->uniformBuffers.at(uniformBuffer).get();
	} catch (const std::out_of_range&) {
		throw Utils::Error("Could not find: %s in 'uniformBuffers'\n", uniformBuffer.c_str());
	}

	return ret;
}

IShaderStorageBuffer* Renderer::getShaderStorageBuffer(const std::string& shaderStorageBuffer) {
	IShaderStorageBuffer* ret = nullptr;

	try {
		ret = this->shaderStorageBuffers.at(shaderStorageBuffer).get();
	} catch (const std::out_of_range&) {
		throw Utils::Error("Could not find: %s in 'shaderStorageBuffers'\n", shaderStorageBuffer.c_str());
	}

	return ret;
}

DescriptorSet* Renderer::getDescriptorSet(const std::string& descriptorSet) {
	DescriptorSet* ret = nullptr;

	try {
		ret = this->descriptorSets.at(descriptorSet).get();
	} catch (const std::out_of_range&) {
		throw Utils::Error("Could not find: %s in 'descriptorSets'\n", descriptorSet.c_str());
	}

	return ret;
}

std::vector<std::pair<std::string, _PostProcessingEffect>>& Renderer::getPostProcessingEffects() {
	return this->postProcessingEffects;
}

vk::Sampler& Renderer::getDefaultSampler() {
	return this->defaultSampler;
}

std::uint32_t Renderer::getFrameIndex() {
	return this->frameIndex;
}

std::uint32_t Renderer::getImageIndex() {
	return this->imageIndex;
}

Uniforms& Renderer::getUniforms() {
	return this->uniforms;
}

SSBOs& Renderer::getSSBOs() {
	return this->ssbos;
}

int& Renderer::getRenderingType() {
	return this->renderingType;
}

bool& Renderer::getShadowsEnabled() {
	return this->shadowsEnabled;
}

float& Renderer::getDepthBiasConstant() {
	return this->depthBiasConstant;
}

float& Renderer::getDepthBiasSlopeFactor() {
	return this->depthBiasSlopeFactor;
}

bool& Renderer::getDebugView() {
	return this->debugView;
}

int& Renderer::getDebugState() {
	return this->debugState;
}

bool& Renderer::getMosaicEnabled() {
	return this->mosaicEnabled;
}

std::pair<vk::Image, vk::ImageView>& Renderer::getDummyTexture() {
	return this->dummyTexture;
}

void Renderer::setRecreateSwapchain(bool value, bool force) {
	this->recreateSwapchain = value;
	this->forceRecreate = force;
}
