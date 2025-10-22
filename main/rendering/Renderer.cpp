#include "Renderer.hpp"

#include <iostream>
#include <random>

#include "Error.hpp"
#include "toString.hpp"

#include "../Driver.hpp"
#include "../baked/BakedModel.hpp"
#include "../baked/BakedModelLoader.hpp"

#include "PipelineCreation.hpp"
#include "RendererObjects.hpp"

#include "../vulkan/Swapchain.hpp"
#include "../vulkan/VulkanDevice.hpp"
#include "../vulkan/VkUtils.hpp"
#include "RendererUtils.hpp"

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define SSAO_KERNEL_SIZE 32

std::uniform_real_distribution<float> randomFloats(0.0f, 1.0f);
std::default_random_engine randomEngine;

Renderer::Renderer(Driver* driver) : driver(driver) {
	this->context.window = std::make_unique<VulkanWindow>();
	this->context.allocator = std::make_unique<VulkanAllocator>(this->context.window.get());

	this->createDummyTexture();

	VulkanWindow* window = this->context.window.get();
	VulkanAllocator* allocator = this->context.allocator.get();
	VulkanDevice* device = window->getDevice();
	Swapchain* swapchain = window->getSwapchain();

	// Create camera in-place
	this->camera = std::make_unique<Camera>(swapchain, 90.0f, 0.01f, 256.0f, glm::vec3(0.0f, 7.0f, -12.0f), glm::vec3(0.0f, 0.0f, -1.0f));

	std::array<const char*, 6> skyboxPaths = {
		"assets-src/main/skybox/right.bmp",
		"assets-src/main/skybox/left.bmp",
		"assets-src/main/skybox/top.bmp",
		"assets-src/main/skybox/bottom.bmp",
		"assets-src/main/skybox/front.bmp",
		"assets-src/main/skybox/back.bmp"
	};

	this->getSkyboxDimensions(skyboxPaths);

	// Note: I no longer like this idea of creating a whole new class for each
	// vulkan object, it seemed like a good idea initially, but the more I add,
	// the more chaotic and unnecessary it seems. Hopefully I get around to reworking it...

	// Render passes
	this->renderPasses.emplace("forward", std::make_unique<ForwardPass>(window, &this->sampleCountSetting));
	this->renderPasses.emplace("deferredWriting", std::make_unique<DeferredWritingPass>(window, &this->sampleCountSetting));
	this->renderPasses.emplace("deferredShading", std::make_unique<DeferredShadingPass>(window, &this->sampleCountSetting));
	this->renderPasses.emplace("shadow", std::make_unique<ShadowPass>(window, &this->sampleCountSetting));
	this->renderPasses.emplace("gui", std::make_unique<GUIPass>(window, &this->sampleCountSetting));
	this->renderPasses.emplace("sunView", std::make_unique<SunViewPass>(window, &this->sampleCountSetting));
	this->renderPasses.emplace("postProcessHDR", std::make_unique<PostProcessHDRPass>(window));
	this->renderPasses.emplace("postProcessLDR", std::make_unique<PostProcessLDRPass>(window));
	this->renderPasses.emplace("tonemap", std::make_unique<TonemapPass>(window));
	this->renderPasses.emplace("pre_ssao", std::make_unique<PreSSAOPass>(window));
	this->renderPasses.emplace("ssao", std::make_unique<SSAOPass>(window));
	this->renderPasses.emplace("debug", std::make_unique<DebugPass>(window));
	this->renderPasses.emplace("skybox", std::make_unique<SkyboxPass>(window));
	this->renderPasses.emplace("sun", std::make_unique<SunPass>(window));
	this->renderPasses.emplace("varianceShadow", std::make_unique<VarianceShadowPass>(window, &this->sampleCountSetting));
	this->renderPasses.emplace("varianceBlur", std::make_unique<VarianceShadowBlurPass>(window));

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
	std::vector<DescriptorSetting> deferredInputs = {
		{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT },
		{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT },
		{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT },
		{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT } };
	std::vector<DescriptorSetting> ssaoTextures = {
		{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT },
		{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT },
		{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT } };

	this->descriptorSetLayouts.emplace("uboV", createDescriptorLayout(*device, uniformBufferV));
	this->descriptorSetLayouts.emplace("uboF", createDescriptorLayout(*device, uniformBufferF));
	this->descriptorSetLayouts.emplace("uboVF", createDescriptorLayout(*device, uniformBufferVF));
	this->descriptorSetLayouts.emplace("ssboF", createDescriptorLayout(*device, ssboF));
	this->descriptorSetLayouts.emplace("imageF", createDescriptorLayout(*device, imageF));
	this->descriptorSetLayouts.emplace("materials", createDescriptorLayout(*device, materialSettings));
	this->descriptorSetLayouts.emplace("deferredInputs", createDescriptorLayout(*device, deferredInputs));
	this->descriptorSetLayouts.emplace("ssaoTextures", createDescriptorLayout(*device, ssaoTextures));

	// Pipeline Layouts
	this->pipelineLayouts.emplace("shadow", std::make_unique<ShadowPipelineLayout>(window, &this->descriptorSetLayouts));
	this->pipelineLayouts.emplace("cubemapShadow", std::make_unique<CubemapShadowPipelineLayout>(window, &this->descriptorSetLayouts));
	this->pipelineLayouts.emplace("varianceShadow", std::make_unique<VarianceShadowPipelineLayout>(window, &this->descriptorSetLayouts));
	this->pipelineLayouts.emplace("forward", std::make_unique<ForwardPipelineLayout>(window, &this->descriptorSetLayouts, &this->shadowsEnabled));
	this->pipelineLayouts.emplace("deferredWriting", std::make_unique<DeferredWritingPipelineLayout>(window, &this->descriptorSetLayouts));
	this->pipelineLayouts.emplace("deferredShading", std::make_unique<DeferredShadingPipelineLayout>(window, &this->descriptorSetLayouts, &this->shadowsEnabled));
	this->pipelineLayouts.emplace("bloom", std::make_unique<BloomPipelineLayout>(window, &this->descriptorSetLayouts));
	this->pipelineLayouts.emplace("mosaic", std::make_unique<MosaicPipelineLayout>(window, &this->descriptorSetLayouts));
	this->pipelineLayouts.emplace("composition", std::make_unique<CompositionPipelineLayout>(window, &this->descriptorSetLayouts));
	this->pipelineLayouts.emplace("tonemap", std::make_unique<TonemapPipelineLayout>(window, &this->descriptorSetLayouts));
	this->pipelineLayouts.emplace("pre_ssao", std::make_unique<PreSSAOPipelineLayout>(window, &this->descriptorSetLayouts));
	this->pipelineLayouts.emplace("ssao", std::make_unique<SSAOPipelineLayout>(window, &this->descriptorSetLayouts));
	this->pipelineLayouts.emplace("ssao_blur", std::make_unique<SSAOBlurPipelineLayout>(window, &this->descriptorSetLayouts));
	this->pipelineLayouts.emplace("skybox", std::make_unique<SkyboxPipelineLayout>(window, &this->descriptorSetLayouts));
	this->pipelineLayouts.emplace("sun", std::make_unique<SunPipelineLayout>(window, &this->descriptorSetLayouts));
	
	// Debug pipeline layouts
	this->pipelineLayouts.emplace("sunView", std::make_unique<SunViewPipelineLayout>(window, &this->descriptorSetLayouts));
	this->pipelineLayouts.emplace("lineDebug", std::make_unique<LineDebugPipelineLayout>(window, &this->descriptorSetLayouts));
	this->pipelineLayouts.emplace("debugViews", std::make_unique<DebugViewsPipelineLayout>(window, &this->descriptorSetLayouts));
	this->pipelineLayouts.emplace("overVisualisation", std::make_unique<OverVisualisationPipelineLayout>(window, &this->descriptorSetLayouts));

	// Pipelines
	// (alot of the pipelines are identical other than their input pipeline layout, renderpass, and shader stages, possibly use the base class for these?)
	// (cubemap)shadow - shadow pass stage pipelines
	this->pipelines.emplace("shadowDirectional", std::make_unique<ShadowPipeline>(window, this->getPipelineLayout("shadow"), this->getRenderPass("shadow"), &this->sampleCountSetting, &this->sunShadowMapRes));
	this->pipelines.emplace("shadowSpot", std::make_unique<ShadowPipeline>(window, this->getPipelineLayout("shadow"), this->getRenderPass("shadow"), &this->sampleCountSetting, &this->spotShadowMapRes));
	this->pipelines.emplace("cubemapShadow", std::make_unique<CubemapShadowPipeline>(window, this->getPipelineLayout("cubemapShadow"), this->getRenderPass("shadow"), &this->sampleCountSetting, &this->pointShadowMapRes));
	this->pipelines.emplace("varianceShadow", std::make_unique<VarianceShadowPipeline>(window, this->getPipelineLayout("varianceShadow"), this->getRenderPass("varianceShadow"), &this->sampleCountSetting, &this->sunShadowMapRes));
	// forward - regular forward shading pipeline
	this->pipelines.emplace("forward", std::make_unique<ForwardPipeline>(window, this->getPipelineLayout("forward"), this->getRenderPass("forward"), &this->sampleCountSetting, &this->shadowsEnabled));
	// deferredWriting - pipeline stage for writing to g-buffers
	this->pipelines.emplace("deferredWriting", std::make_unique<DeferredWritingPipeline>(window, this->getPipelineLayout("deferredWriting"), this->getRenderPass("deferredWriting"), &this->sampleCountSetting, &this->ssaoEnabled));
	// deferredShading - pipeline stage for shading pass in deferred rendering
	this->pipelines.emplace("deferredShading", std::make_unique<DeferredShadingPipeline>(window, this->getPipelineLayout("deferredShading"), this->getRenderPass("deferredShading"), &this->sampleCountSetting, &this->shadowsEnabled, &this->ssaoEnabled));
	// post processing effects
	this->pipelines.emplace("mosaic", std::make_unique<MosaicPipeline>(window, this->getPipelineLayout("mosaic"), this->getRenderPass("postProcessLDR")));
	this->pipelines.emplace("bloom", std::make_unique<BlurPipeline>(window, this->getPipelineLayout("bloom"), this->getRenderPass("postProcessHDR"), &this->bloomTapSize));
	this->pipelines.emplace("varianceBlur", std::make_unique<BlurPipeline>(window, this->getPipelineLayout("bloom"), this->getRenderPass("varianceBlur"), &this->vsmTapSize, &this->sunShadowMapRes));
	this->pipelines.emplace("composition", std::make_unique<CompositionPipeline>(window, this->getPipelineLayout("composition"), this->getRenderPass("postProcessHDR")));
	this->pipelines.emplace("tonemap", std::make_unique<TonemapPipeline>(window, this->getPipelineLayout("tonemap"), this->getRenderPass("postProcessLDR")));
	this->pipelines.emplace("fxaa", std::make_unique<FXAAPipeline>(window, this->getPipelineLayout("mosaic"), this->getRenderPass("postProcessLDR")));
	this->pipelines.emplace("pre_ssao", std::make_unique<PreSSAOPipeline>(window, this->getPipelineLayout("pre_ssao"), this->getRenderPass("pre_ssao")));
	this->pipelines.emplace("ssao", std::make_unique<SSAOPipeline>(window, this->getPipelineLayout("ssao"), this->getRenderPass("ssao")));
	this->pipelines.emplace("ssao_blur", std::make_unique<SSAOBlurPipeline>(window, this->getPipelineLayout("ssao_blur"), this->getRenderPass("ssao")));
	this->pipelines.emplace("skybox", std::make_unique<SkyboxPipeline>(window, this->getPipelineLayout("skybox"), this->getRenderPass("postProcessHDR")));
	this->pipelines.emplace("sun", std::make_unique<SunPipeline>(window, this->getPipelineLayout("sun"), this->getRenderPass("sun")));

	// Debug pipelines
	this->pipelines.emplace("sunView", std::make_unique<SunViewPipeline>(window, this->getPipelineLayout("sunView"), this->getRenderPass("sunView"), &this->sampleCountSetting));
	this->pipelines.emplace("lineDebug", std::make_unique<LineDebugPipeline>(window, this->getPipelineLayout("lineDebug"), this->getRenderPass("sunView"), &this->sampleCountSetting));
	this->pipelines.emplace("debugViews", std::make_unique<DebugViewsPipeline>(window, this->getPipelineLayout("debugViews"), this->getRenderPass("debug"), &this->sampleCountSetting));
	this->pipelines.emplace("overVisualisation", std::make_unique<OverVisualisationPipeline>(window, this->getPipelineLayout("overVisualisation"), this->getRenderPass("debug"), &this->sampleCountSetting));

	// Texture Buffers
	// HDR - output buffer after geometry and lighting (HDR rendering)
	// LDR - output to be used after tonemapping and subsequent post processing effects
	this->textureBuffers.emplace("HDR", std::make_unique<ColourTextureBuffer>(&this->context));
	this->textureBuffers.emplace("LDR", std::make_unique<ColourTextureBuffer>(&this->context, VK_FORMAT_R8G8B8A8_UNORM));
	// brightness - buffer used to render scene brightness as input for bloom post processing
	this->textureBuffers.emplace("brightness", std::make_unique<ColourTextureBuffer>(&this->context));
	// intermediate - intermediate buffers used during post processing effects
	this->textureBuffers.emplace("intermediateHDR", std::make_unique<ColourTextureBuffer>(&this->context));
	this->textureBuffers.emplace("intermediateHDR2", std::make_unique<ColourTextureBuffer>(&this->context));
	this->textureBuffers.emplace("intermediateLDR", std::make_unique<ColourTextureBuffer>(&this->context, VK_FORMAT_R8G8B8A8_UNORM));
	// depth - standard depth buffer
	this->textureBuffers.emplace("depth", std::make_unique<DepthTextureBuffer>(&this->context));
	// gBuffers - g-buffers used in deferred rendering
	this->textureBuffers.emplace("gBuffer1", std::make_unique<ColourTextureBuffer>(&this->context, VK_FORMAT_A2R10G10B10_UNORM_PACK32));
	this->textureBuffers.emplace("gBuffer2", std::make_unique<ColourTextureBuffer>(&this->context, VK_FORMAT_R8G8B8A8_UNORM));
	this->textureBuffers.emplace("gBuffer3", std::make_unique<ColourTextureBuffer>(&this->context, VK_FORMAT_R8G8B8A8_UNORM));
	// blurOutput - output of blur in bloom post processing
	this->textureBuffers.emplace("blurOutput", std::make_unique<ColourTextureBuffer>(&this->context));
	this->textureBuffers.emplace("varianceBlurOutput", std::make_unique<ColourTextureBuffer>(&this->context, VK_FORMAT_R32G32_SFLOAT, nullptr, &this->sunShadowMapRes));
	// noise - used in SSAO
	this->textureBuffers.emplace("noise", std::make_unique<NoiseTextureBuffer>(&this->context));
	// ssao - result of SSAO pass
	this->textureBuffers.emplace("ssao", std::make_unique<ColourTextureBuffer>(&this->context, VK_FORMAT_R8_UNORM, nullptr, &swapchain->getHalfExtent()));
	this->textureBuffers.emplace("ssaoHblur", std::make_unique<ColourTextureBuffer>(&this->context, VK_FORMAT_R8_UNORM));
	this->textureBuffers.emplace("ssaoVblur", std::make_unique<ColourTextureBuffer>(&this->context, VK_FORMAT_R8_UNORM));
	// skybox - skybox cubemap
	this->textureBuffers.emplace("skybox", std::make_unique<CubemapTextureBuffer>(&this->context, VK_FORMAT_R8G8B8A8_SRGB, &this->skyboxDimensions, true, true));

	// Debug texture buffers
	// sunView - buffer used to render the sun view
	this->textureBuffers.emplace("sunView", std::make_unique<ColourTextureBuffer>(&this->context));

	// Framebuffers
	// forward - 1 colour, 1 depth render targets
	this->framebuffers.emplace("forward", std::make_unique<ForwardFramebuffer>(window, &this->textureBuffers, this->getRenderPass("forward"), &this->sampleCountSetting));
	// deferred - 4 colour (3 g-buffers, 1 colour), 1 depth render targets
	this->framebuffers.emplace("deferredWriting", std::make_unique<DeferredWritingFramebuffer>(window, &this->textureBuffers, this->getRenderPass("deferredWriting"), &this->sampleCountSetting));
	this->framebuffers.emplace("deferredShading", std::make_unique<DeferredShadingFramebuffer>(window, &this->textureBuffers, this->getRenderPass("deferredShading"), &this->sampleCountSetting));
	// sun - 1 colour (sunView buffer), 1 depth render targets
	this->framebuffers.emplace("sunView", std::make_unique<SunFramebuffer>(window, &this->textureBuffers, this->getRenderPass("sunView"), &this->sampleCountSetting));
	// gui - writes directly to swapchain buffer
	this->framebuffers.emplace("gui", std::make_unique<GUIFramebuffer>(window, this->getRenderPass("gui")));
	// writeTo - framebuffers used to write to a single render target, usually used to ping-pong writing to buffers during post processing
	this->framebuffers.emplace("writeToHDR", std::make_unique<WriteToTargetFramebuffer>(window, this->getTextureBuffer("HDR"), this->getRenderPass("postProcessHDR")));
	this->framebuffers.emplace("writeToLDR", std::make_unique<WriteToTargetFramebuffer>(window, this->getTextureBuffer("LDR"), this->getRenderPass("postProcessLDR")));
	this->framebuffers.emplace("writeToIntermediateHDR", std::make_unique<WriteToTargetFramebuffer>(window, this->getTextureBuffer("intermediateHDR"), this->getRenderPass("postProcessHDR")));
	this->framebuffers.emplace("writeToIntermediateHDR2", std::make_unique<WriteToTargetFramebuffer>(window, this->getTextureBuffer("intermediateHDR2"), this->getRenderPass("postProcessHDR")));
	this->framebuffers.emplace("writeToIntermediateLDR", std::make_unique<WriteToTargetFramebuffer>(window, this->getTextureBuffer("intermediateLDR"), this->getRenderPass("postProcessLDR")));
	this->framebuffers.emplace("writeToBlur", std::make_unique<WriteToTargetFramebuffer>(window, this->getTextureBuffer("blurOutput"), this->getRenderPass("postProcessHDR")));
	this->framebuffers.emplace("writeToVarianceBlur", std::make_unique<WriteToTargetFramebuffer>(window, this->getTextureBuffer("varianceBlurOutput"), this->getRenderPass("varianceBlur"), &this->sunShadowMapRes));
	// pre_ssao - writes to normal gbuffer and depth
	this->framebuffers.emplace("pre_ssao", std::make_unique<PreSSAOFramebuffer>(window, &this->textureBuffers, this->getRenderPass("pre_ssao")));
	// ssao - write to ssao texture buffer after SSAO pass
	this->framebuffers.emplace("ssao", std::make_unique<WriteToTargetFramebuffer>(window, this->getTextureBuffer("ssao"), this->getRenderPass("ssao"), &swapchain->getHalfExtent()));
	this->framebuffers.emplace("ssaoHblur", std::make_unique<WriteToTargetFramebuffer>(window, this->getTextureBuffer("ssaoHblur"), this->getRenderPass("ssao")));
	this->framebuffers.emplace("ssaoVblur", std::make_unique<WriteToTargetFramebuffer>(window, this->getTextureBuffer("ssaoVblur"), this->getRenderPass("ssao")));
	this->framebuffers.emplace("debug", std::make_unique<DebugFramebuffer>(window, &this->textureBuffers, this->getRenderPass("debug")));

	// Uniform Buffers
	VkPipelineStageFlags VFstageFlags = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	VkPipelineStageFlags VstageFlags = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT;
	VkPipelineStageFlags FstageFlags = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

	this->uniformBuffers.emplace("mvp", std::make_unique<UniformBuffer<glsl::MVPUniform>>(&this->context, VFstageFlags, &this->uniforms.mvpUniform));
	this->uniformBuffers.emplace("cameraPlanes", std::make_unique<UniformBuffer<glsl::CameraPlanesUniform>>(&this->context, FstageFlags, &this->uniforms.cameraPlanesUniform));
	this->uniformBuffers.emplace("projections", std::make_unique<UniformBuffer<glsl::ProjectiveUniform>>(&this->context, FstageFlags, &this->uniforms.projectiveUniform));
	this->uniformBuffers.emplace("invMatrices", std::make_unique<UniformBuffer<glsl::InverseMatricesUniform>>(&this->context, FstageFlags, &this->uniforms.inverseMatricesUniform));
	this->uniformBuffers.emplace("ssao", std::make_unique<UniformBuffer<glsl::SSAOUniform>>(&this->context, FstageFlags, &this->uniforms.ssaoUniform));

	// Synchronisation
	for (std::size_t i = 0; i < Swapchain::MAX_FRAMES_IN_FLIGHT; i++) {
		this->cmdBuffers.emplace_back(VkUtils::createCommandBuffer(*window, device->getCmdPool()));
		this->frameDoneFences.emplace_back(VkUtils::createFence(*window, VK_FENCE_CREATE_SIGNALED_BIT));
		this->imageAvailableSemaphores.emplace_back(VkUtils::createSemaphore(*window));
		this->renderFinishedSemaphores.emplace_back(VkUtils::createSemaphore(*window));
	}

	// Samplers
	SamplerInfo linearRepeatSamplerInfo = {
		.magFilter = VK_FILTER_LINEAR,
		.minFilter = VK_FILTER_LINEAR,
		.addressMode = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		/*.anisotropyEnable = VK_TRUE*/ };
	this->linearRepeatSampler = VkUtils::createTextureSampler(*window, linearRepeatSamplerInfo);
	SamplerInfo linearMirroredRepeatSamplerInfo = {
		.magFilter = VK_FILTER_LINEAR,
		.minFilter = VK_FILTER_LINEAR,
		.addressMode = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT };
	this->linearMirroredRepeatSampler = VkUtils::createTextureSampler(*window, linearMirroredRepeatSamplerInfo);
	SamplerInfo linearClampToEdgeSamplerInfo = {
		.magFilter = VK_FILTER_LINEAR,
		.minFilter = VK_FILTER_LINEAR,
		.addressMode = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE };
	this->linearClampToEdgeSampler = VkUtils::createTextureSampler(*window, linearClampToEdgeSamplerInfo);
	SamplerInfo linearClampToBorderSamplerInfo = {
		.magFilter = VK_FILTER_LINEAR,
		.minFilter = VK_FILTER_LINEAR,
		.addressMode = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
		.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE };
	this->linearClampToBorderSampler = VkUtils::createTextureSampler(*window, linearClampToBorderSamplerInfo);
	SamplerInfo nearestRepeatSamplerInfo = {
		.magFilter = VK_FILTER_NEAREST,
		.minFilter = VK_FILTER_NEAREST,
		.addressMode = VK_SAMPLER_ADDRESS_MODE_REPEAT };
	this->nearestRepeatSampler = VkUtils::createTextureSampler(*window, nearestRepeatSamplerInfo);
	SamplerInfo nearestClampToEdgeSamplerInfo = {
		.magFilter = VK_FILTER_NEAREST,
		.minFilter = VK_FILTER_NEAREST,
		.addressMode = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE };
	this->nearestClampToEdgeSampler = VkUtils::createTextureSampler(*window, nearestClampToEdgeSamplerInfo);
	SamplerInfo depthSamplerInfo = {
		.magFilter = VK_FILTER_LINEAR,
		.minFilter = VK_FILTER_LINEAR,
		.addressMode = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER };
	this->depthSampler = VkUtils::createTextureSampler(*window, depthSamplerInfo);
	SamplerInfo shadowMapSamplerInfo = {
		.magFilter = VK_FILTER_LINEAR,
		.minFilter = VK_FILTER_LINEAR,
		.addressMode = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
		.compareEnable = 1, 
		.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
		.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE };
	this->shadowMapSampler = VkUtils::createTextureSampler(*window, shadowMapSamplerInfo);
	

	// Descriptor Sets
	// (this looks unbearably messy, surely there is a nicer way of defining what makes up a descriptor
	// set than this)
	std::vector<DescriptorBufferSetting> mvpDescriptorSettings = {
		{ this->getUniformBuffer("mvp"), VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER } };
	std::vector<DescriptorBufferSetting> cameraPlanesDescriptorSettings = {
		{ this->getUniformBuffer("cameraPlanes"), VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER } };
	std::vector<DescriptorImageSetting> deferredInputsDescriptorSettings = {
		{ this->getTextureBuffer("gBuffer1"), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, this->linearClampToEdgeSampler.handle },
		{ this->getTextureBuffer("gBuffer2"), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, this->linearClampToEdgeSampler.handle },
		{ this->getTextureBuffer("gBuffer3"), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, this->linearClampToEdgeSampler.handle },
		{ this->getTextureBuffer("depth"), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, this->linearClampToEdgeSampler.handle } };
	std::vector<DescriptorImageSetting> sunViewDescriptorSettings = {
		{ this->getTextureBuffer("sunView"), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, this->linearRepeatSampler.handle } };
	std::vector<DescriptorImageSetting> HDROuputDescriptorSettings = {
		{ this->getTextureBuffer("HDR"), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, this->linearRepeatSampler.handle } };
	std::vector<DescriptorImageSetting> LDROuputDescriptorSettings = {
		{ this->getTextureBuffer("LDR"), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, this->linearRepeatSampler.handle } };
	std::vector<DescriptorImageSetting> intermediateHDRImageDescriptorSettings = {
		{ this->getTextureBuffer("intermediateHDR"), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, this->linearRepeatSampler.handle } };
	std::vector<DescriptorImageSetting> intermediateHDR2ImageDescriptorSettings = {
		{ this->getTextureBuffer("intermediateHDR2"), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, this->linearClampToEdgeSampler.handle } };
	std::vector<DescriptorImageSetting> intermediateLDRImageDescriptorSettings = {
		{ this->getTextureBuffer("intermediateLDR"), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, this->linearRepeatSampler.handle } };
	std::vector<DescriptorImageSetting> brightnessImageDescriptorSettings = {
		{ this->getTextureBuffer("brightness"), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, this->linearClampToEdgeSampler.handle } };
	std::vector<DescriptorImageSetting> blurImageDescriptorSettings = {
		{ this->getTextureBuffer("blurOutput"), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, this->linearClampToEdgeSampler.handle } };
	std::vector<DescriptorBufferSetting> projectionDescriptorSettings = {
		{ this->getUniformBuffer("projections"), VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER } };
	std::vector<DescriptorBufferSetting> inverseMatricesDescriptorSettings = {
		{ this->getUniformBuffer("invMatrices"), VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER } };
	std::vector<DescriptorBufferSetting> ssaoDescriptorSettings = {
		{ this->getUniformBuffer("ssao"), VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER } };
	std::vector<DescriptorImageSetting> ssaoTexturesDescriptorSettings = {
		{ this->getTextureBuffer("depth"), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, this->linearMirroredRepeatSampler.handle },
		{ this->getTextureBuffer("gBuffer1"), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, this->nearestClampToEdgeSampler.handle },
		{ this->getTextureBuffer("noise"), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, this->nearestRepeatSampler.handle } };
	std::vector<DescriptorImageSetting> ssaoHblurDescriptorSettings = {
		{ this->getTextureBuffer("depth"), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, this->linearMirroredRepeatSampler.handle },
		{ this->getTextureBuffer("gBuffer1"), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, this->nearestClampToEdgeSampler.handle },
		{ this->getTextureBuffer("ssao"), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, this->nearestClampToEdgeSampler.handle }  };
	std::vector<DescriptorImageSetting> ssaoVblurDescriptorSettings = {
		{ this->getTextureBuffer("depth"), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, this->linearMirroredRepeatSampler.handle },
		{ this->getTextureBuffer("gBuffer1"), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, this->nearestClampToEdgeSampler.handle },
		{ this->getTextureBuffer("ssaoHblur"), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, this->nearestClampToEdgeSampler.handle } };
	std::vector<DescriptorImageSetting> ssaoSamplerDescriptorSettings = {
		{ this->getTextureBuffer("ssaoVblur"), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, this->nearestClampToEdgeSampler.handle } };
	std::vector<DescriptorImageSetting> skyboxDescriptorSettings = {
		{ this->getTextureBuffer("skybox"), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, this->linearRepeatSampler.handle } };
	std::vector<DescriptorImageSetting> varianceBlurDescriptorSettings = {
		{ this->getTextureBuffer("varianceBlurOutput"), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, this->linearClampToBorderSampler.handle } };

	this->descriptorSets.emplace("mvp", std::make_unique<BufferDescriptorSet>(window, &this->descriptorSetLayouts.at("uboVF").handle, mvpDescriptorSettings));
	this->descriptorSets.emplace("cameraPlanes", std::make_unique<BufferDescriptorSet>(window, &this->descriptorSetLayouts.at("uboF").handle, cameraPlanesDescriptorSettings));
	this->descriptorSets.emplace("deferredInputs", std::make_unique<ImageDescriptorSet>(window, &this->descriptorSetLayouts.at("deferredInputs").handle, deferredInputsDescriptorSettings));
	this->descriptorSets.emplace("sunView", std::make_unique<ImageDescriptorSet>(window, &this->descriptorSetLayouts.at("imageF").handle, sunViewDescriptorSettings));
	this->descriptorSets.emplace("HDR", std::make_unique<ImageDescriptorSet>(window, &this->descriptorSetLayouts.at("imageF").handle, HDROuputDescriptorSettings));
	this->descriptorSets.emplace("LDR", std::make_unique<ImageDescriptorSet>(window, &this->descriptorSetLayouts.at("imageF").handle, LDROuputDescriptorSettings));
	this->descriptorSets.emplace("intermediateHDR", std::make_unique<ImageDescriptorSet>(window, &this->descriptorSetLayouts.at("imageF").handle, intermediateHDRImageDescriptorSettings));
	this->descriptorSets.emplace("intermediateHDR2", std::make_unique<ImageDescriptorSet>(window, &this->descriptorSetLayouts.at("imageF").handle, intermediateHDR2ImageDescriptorSettings));
	this->descriptorSets.emplace("intermediateLDR", std::make_unique<ImageDescriptorSet>(window, &this->descriptorSetLayouts.at("imageF").handle, intermediateLDRImageDescriptorSettings));
	this->descriptorSets.emplace("brightness", std::make_unique<ImageDescriptorSet>(window, &this->descriptorSetLayouts.at("imageF").handle, brightnessImageDescriptorSettings));
	this->descriptorSets.emplace("blurOutput", std::make_unique<ImageDescriptorSet>(window, &this->descriptorSetLayouts.at("imageF").handle, blurImageDescriptorSettings));
	this->descriptorSets.emplace("projections", std::make_unique<BufferDescriptorSet>(window, &this->descriptorSetLayouts.at("uboF").handle, projectionDescriptorSettings));
	this->descriptorSets.emplace("invMatrices", std::make_unique<BufferDescriptorSet>(window, &this->descriptorSetLayouts.at("uboF").handle, inverseMatricesDescriptorSettings));
	this->descriptorSets.emplace("ssao", std::make_unique<BufferDescriptorSet>(window, &this->descriptorSetLayouts.at("uboF").handle, ssaoDescriptorSettings));
	this->descriptorSets.emplace("ssaoTextures", std::make_unique<ImageDescriptorSet>(window, &this->descriptorSetLayouts.at("ssaoTextures").handle, ssaoTexturesDescriptorSettings));
	this->descriptorSets.emplace("ssaoHBlurTextures", std::make_unique<ImageDescriptorSet>(window, &this->descriptorSetLayouts.at("ssaoTextures").handle, ssaoHblurDescriptorSettings));
	this->descriptorSets.emplace("ssaoVBlurTextures", std::make_unique<ImageDescriptorSet>(window, &this->descriptorSetLayouts.at("ssaoTextures").handle, ssaoVblurDescriptorSettings));
	this->descriptorSets.emplace("ssaoSampler", std::make_unique<ImageDescriptorSet>(window, &this->descriptorSetLayouts.at("imageF").handle, ssaoSamplerDescriptorSettings));
	this->descriptorSets.emplace("skybox", std::make_unique<ImageDescriptorSet>(window, &this->descriptorSetLayouts.at("imageF").handle, skyboxDescriptorSettings));
	this->descriptorSets.emplace("varianceBlur", std::make_unique<ImageDescriptorSet>(window, &this->descriptorSetLayouts.at("imageF").handle, varianceBlurDescriptorSettings));

	// Pre processing effects
	// (pre processing effects are what I call effects / passes in the renderer
	// that affect the main scene rendering / lighting by doing something beforehand,
	// i.e. ssao)
	this->ssaoEffect = std::make_unique<SSAOPreProcess>(this);

	// Post processing effects
	this->postProcessingEffects.emplace_back(std::make_pair("bloom", std::make_unique<BloomPostProcess>(this)));
	this->postProcessingEffects.emplace_back(std::make_pair("tonemap", std::make_unique<TonemapPostProcess>(this)));
	this->postProcessingEffects.emplace_back(std::make_pair("fxaa", std::make_unique<FXAAPostProcess>(this)));
	this->postProcessingEffects.emplace_back(std::make_pair("mosaic", std::make_unique<MosaicPostProcess>(this)));

	// SSAO uniform data only needs to be initialised once
	for (int i = 0; i < SSAO_KERNEL_SIZE; i++) {
		glm::vec3 sample(
			randomFloats(randomEngine) * 2.0f - 1.0f,
			randomFloats(randomEngine) * 2.0f - 1.0f,
			randomFloats(randomEngine)); // Z-axis only goes [0, 1] otherwise kernel would be sphere, we want hemisphere
		sample = glm::normalize(sample);
		sample *= randomFloats(randomEngine);

		float scale = float(i) / float(SSAO_KERNEL_SIZE);
		scale = std::lerp(0.1f, 1.0f, scale * scale);
		this->uniforms.ssaoUniform.samples[i] = glm::vec4(sample * scale, 0.0f);
	}
	this->uniforms.ssaoUniform.radius = 0.5f;

	// Fill skybox texture (data already gathered from getSkyboxDimensions())
	this->fillSkyboxTexture();
}

Renderer::~Renderer() {
	// Used in case Vulkan/VMA throws an error during rendering,
	// we need to manually destroy ImGUI related Vulkan objects
	// through its shutdown methods before destroying our Vulkan 
	// device instance
	if (!this->handledImGUIShutdown) {
		vkDeviceWaitIdle(this->context.window->getDevice()->getDevice());
		RendererUtils::destroyImGUI();
	}
}

void Renderer::getSkyboxDimensions(std::array<const char*, 6>& filenames) {
	// Load images using stb
	int width, height, channels;

	for (int i = 0; i < 6; i++) {
		this->skyboxImageData[i] = stbi_load(filenames[i], &width, &height, &channels, STBI_rgb_alpha);
		if (!this->skyboxImageData[i])
			throw Utils::Error("Unable to load skybox image: %s\n", filenames[i]);
	}

	this->skyboxDimensions = { std::uint32_t(width), std::uint32_t(height) };
}

void Renderer::fillSkyboxTexture() {
	// Upload images to GPU
	std::size_t imageSize = this->skyboxDimensions.width * this->skyboxDimensions.height * 4;
	std::size_t bufferSize = imageSize * 6;

	vk::Buffer stagingBuffer = vk::createBuffer(
		*this->context.allocator,
		bufferSize,
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

	std::uint8_t* ptr = nullptr;
	if (const auto res = vmaMapMemory(this->context.allocator->allocator, stagingBuffer.allocation, (void**)&ptr); VK_SUCCESS != res)
		throw Utils::Error("Error when mapping memory for writing\nvmaMapMemory() returned %s\n", Utils::toString(res).c_str());

	for (int i = 0; i < 6; i++) {
		// Pointer arithmetic to advance pointer so we can copy the next data of next image
		std::memcpy(ptr + (imageSize * i), skyboxImageData[i], imageSize);
	}

	vmaUnmapMemory(this->context.allocator->allocator, stagingBuffer.allocation);

	// Get skybox VkImage handle
	VkImage image = this->getTextureBuffer("skybox")->getImage().image;

	VkCommandPool cmdPool = this->context.window->getDevice()->getCmdPool();
	VkCommandBuffer cmdBuff = VkUtils::createCommandBuffer(*this->context.window, cmdPool);
	VkUtils::beginCommandBuffer(cmdBuff);

	// Transition to TRANSFER_DST_OPTIMAL
	VkUtils::imageBarrier(cmdBuff, image,
		/* srcAccessMask */ 0, /* dstAccessMask */ VK_ACCESS_TRANSFER_WRITE_BIT,
		/* srcLayout     */ VK_IMAGE_LAYOUT_UNDEFINED, /* dstLayout */ VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		/* srcStageMask  */ VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, /* dstStageMask */ VK_PIPELINE_STAGE_TRANSFER_BIT,
		VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6 });

	// Copy buffer into image
	VkBufferImageCopy copy;
	copy.bufferOffset = 0;
	copy.bufferRowLength = 0;
	copy.bufferImageHeight = 0;
	copy.imageSubresource = VkImageSubresourceLayers{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 6 };
	copy.imageOffset = VkOffset3D{ 0, 0, 0 };
	copy.imageExtent = VkExtent3D{ this->skyboxDimensions.width, this->skyboxDimensions.height, 1 };

	vkCmdCopyBufferToImage(cmdBuff, stagingBuffer.buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

	// Transition to SHADER_READ_ONLY_OPTIMAL
	VkUtils::imageBarrier(cmdBuff, image,
		/* srcAccessMask */ VK_ACCESS_TRANSFER_WRITE_BIT, /* dstAccessMask */ VK_ACCESS_SHADER_READ_BIT,
		/* srcLayout     */ VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, /* dstLayout */ VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		/* srcStageMask  */ VK_PIPELINE_STAGE_TRANSFER_BIT, /* dstStageMask */ VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
		VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6 });

	VkUtils::endAndSubmitCommandBuffer(*this->context.window, cmdBuff);

	for (int i = 0; i < 6; i++) {
		free(this->skyboxImageData[i]);
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
	this->textureBuffers.emplace("pointArrayShadows", std::make_unique<ArrayTextureBuffer>(&this->context, true, this->numPointLights, VK_FORMAT_D32_SFLOAT, &this->pointShadowMapRes));
	this->textureBuffers.emplace("directionalShadow", std::make_unique<ColourTextureBuffer>(&this->context, VK_FORMAT_R32G32_SFLOAT, nullptr, &this->sunShadowMapRes));
	this->textureBuffers.emplace("directionalShadowDepth", std::make_unique<DepthTextureBuffer>(&this->context, VK_FORMAT_D32_SFLOAT, nullptr, &this->sunShadowMapRes));
	this->textureBuffers.emplace("spotArrayShadows",  std::make_unique<ArrayTextureBuffer>(&this->context, false, this->numSpotLights, VK_FORMAT_D32_SFLOAT, &this->spotShadowMapRes));
#ifndef NDEBUG
	// TODO: do we really need RGBA16 formats for these?
	this->textureBuffers.emplace("pointArrayShadowsDebug", std::make_unique<ArrayTextureBuffer>(&this->context, false, this->numPointLights * 6, VK_FORMAT_R16G16B16A16_SFLOAT, &this->pointShadowMapRes));
	this->textureBuffers.emplace("directionalShadowDebug", std::make_unique<ColourTextureBuffer>(&this->context, VK_FORMAT_R16G16B16A16_SFLOAT, nullptr, &this->sunShadowMapRes));
	this->textureBuffers.emplace("spotArrayShadowsDebug",  std::make_unique<ArrayTextureBuffer>(&this->context, false, this->numSpotLights, VK_FORMAT_R16G16B16A16_SFLOAT, &this->spotShadowMapRes));
#endif

	if (this->numPointLights != 0) {
		std::initializer_list<TextureBuffer*> pointShadowTextures = {
			this->getTextureBuffer("pointArrayShadows"),
#ifndef NDEBUG
			this->getTextureBuffer("pointArrayShadowsDebug")
#endif
		};

		this->framebuffers.emplace("pointArrayShadows", std::make_unique<ArrayFramebuffer>(window, pointShadowTextures, this->getRenderPass("shadow"), this->numPointLights * 6, &this->pointShadowMapRes));
	}

	if (this->numDirectionalLights != 0) {
		std::initializer_list<TextureBuffer*> directionalShadowTextures = {
			this->getTextureBuffer("directionalShadow"),
#ifndef NDEBUG
			this->getTextureBuffer("directionalShadowDebug")
#endif
		};

		std::initializer_list<TextureBuffer*> tempTextures = { this->getTextureBuffer("directionalShadow"), this->getTextureBuffer("directionalShadowDepth") };

		//this->framebuffers.emplace("directionalShadow", std::make_unique<ShadowFramebuffer>(window, directionalShadowTextures, this->getRenderPass("shadow"), &this->sunShadowMapRes));
		this->framebuffers.emplace("directionalShadow", std::make_unique<ShadowFramebuffer>(window, tempTextures, this->getRenderPass("varianceShadow"), &this->sunShadowMapRes));
		this->framebuffers.emplace("writeToDirectionalShadow", std::make_unique<WriteToTargetFramebuffer>(window, this->getTextureBuffer("directionalShadow"), this->getRenderPass("varianceBlur"), &this->sunShadowMapRes));
	}

	if (this->numSpotLights != 0) {
		std::initializer_list<TextureBuffer*> spotShadowTextures = {
			this->getTextureBuffer("spotArrayShadows"),
#ifndef NDEBUG
			this->getTextureBuffer("spotArrayShadowsDebug")
#endif
		};

		this->framebuffers.emplace("spotArrayShadows", std::make_unique<ArrayFramebuffer>(window, spotShadowTextures, this->getRenderPass("shadow"), this->numSpotLights, &this->spotShadowMapRes));
	}

	// Light type counters (surely I can make a better system than this)
	int pointLightIndex = 0;
	int spotLightIndex = 0;
	int matrixDependentIndex = 0;

	// Populate ssbos
	for (std::size_t i = 0; i < lights->size(); i++) {
		Light& light = lights->at(i);

		glsl::Light lightStruct = {
			.positionAndLightType = { light.getPosition(), light.getLightType() },
			.directionAndMapIndex = { light.getDirection(), 0 },
			.colourAndIntensity = { light.getColour(), light.getIntensity() },
			.spotLightAndMatrixIndex = { std::cos(glm::radians(light.getInnerAngle())), std::cos(glm::radians(light.getOuterAngle())), 0, 0 }
		};


		switch (light.getLightType()) {
		case LightType::POINT:
		{
			lightStruct.directionAndMapIndex.w = pointLightIndex;
			pointLightIndex++;
			break;
		}
		case LightType::DIRECTIONAL:
		{
			this->sunMatrices.projection = Cache<glm::mat4>([this]() {
				return glm::ortho(-this->sunOrthoBounds, this->sunOrthoBounds, this->sunOrthoBounds, -this->sunOrthoBounds, this->sunShadowNear, this->sunShadowFar);
			});
			this->sunMatrices.view = Cache<glm::mat4>([this, &light]() {
				return glm::lookAt(glm::vec3(0.0f, 0.0f, -50.0f) + (-light.getDirection() * this->sunDistance), glm::vec3(0.0f, 0.0f, -50.0f), glm::vec3(0.0f, 1.0f, 0.0f));
			});

			this->sunLightIndex = i;
			this->ssbos.lightMatrices.emplace_back(this->sunMatrices.projection.get() * this->sunMatrices.view.get());
			this->ssbos.lightMatrices.emplace_back(this->sunMatrices.view.get());

			lightStruct.spotLightAndMatrixIndex.z = matrixDependentIndex;

			matrixDependentIndex++;
			matrixDependentIndex++;
			break;
		}
		case LightType::SPOT:
		{
			lightStruct.directionAndMapIndex.w = spotLightIndex;

			glm::mat4 projection = glm::perspective(glm::radians(90.0f), 1.0f, 0.01f, 256.0f);
			projection[1][1] *= -1.0;
			glm::mat4 view = glm::lookAt(light.getPosition(), light.getPosition() + light.getDirection(), glm::vec3(0.0f, 1.0f, 0.0f));

			this->ssbos.lightMatrices.emplace_back(projection * view);

			lightStruct.spotLightAndMatrixIndex.z = matrixDependentIndex;

			spotLightIndex++;
			matrixDependentIndex++;
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
		{ this->getTextureBuffer("directionalShadow"), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, this->linearClampToBorderSampler.handle } };
	std::vector<DescriptorImageSetting> spotLightShadowsDescriptorSettings = {
		{ this->getTextureBuffer("spotArrayShadows"),  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, this->shadowMapSampler.handle } };

	this->descriptorSets.emplace("pointLightShadows", std::make_unique<ImageDescriptorSet>(window, &this->descriptorSetLayouts.at("imageF").handle, pointLightShadowsDescriptorSettings));
	this->descriptorSets.emplace("directionalLightShadow", std::make_unique<ImageDescriptorSet>(window, &this->descriptorSetLayouts.at("imageF").handle, directionalLightShadowDescriptorSettings));
	this->descriptorSets.emplace("spotLightShadows", std::make_unique<ImageDescriptorSet>(window, &this->descriptorSetLayouts.at("imageF").handle, spotLightShadowsDescriptorSettings));

#ifndef NDEBUG
	std::vector<DescriptorImageSetting> pointLightShadowsDebugDescriptorSettings = {
		{ this->getTextureBuffer("pointArrayShadowsDebug"), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, this->shadowMapSampler.handle } };
	std::vector<DescriptorImageSetting> directionalLightShadowDebugDescriptorSettings = {
		{ this->getTextureBuffer("directionalShadowDebug"), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, this->shadowMapSampler.handle } };
	std::vector<DescriptorImageSetting> spotLightShadowsDebugDescriptorSettings = {
		{ this->getTextureBuffer("spotArrayShadowsDebug"),  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, this->shadowMapSampler.handle } };

	this->descriptorSets.emplace("pointLightShadowsDebug", std::make_unique<ArrayImageDescriptorSet>(window, &this->descriptorSetLayouts.at("imageF").handle, pointLightShadowsDebugDescriptorSettings));
	this->descriptorSets.emplace("directionalLightShadowDebug", std::make_unique<ImageDescriptorSet>(window, &this->descriptorSetLayouts.at("imageF").handle, directionalLightShadowDebugDescriptorSettings));
	this->descriptorSets.emplace("spotLightShadowsDebug", std::make_unique<ArrayImageDescriptorSet>(window, &this->descriptorSetLayouts.at("imageF").handle, spotLightShadowsDebugDescriptorSettings));
#endif

	std::vector<DescriptorBufferSetting> lightSSBODescriptorSettings = {
		{ this->getShaderStorageBuffer("lights"), VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, this->getShaderStorageBuffer("lights")->getBufferSize() } };
	std::vector<DescriptorBufferSetting> lightMatricesSSBODescriptorSettings = {
		{ this->getShaderStorageBuffer("lightMatrices"), VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, this->getShaderStorageBuffer("lightMatrices")->getBufferSize() } };

	this->descriptorSets.emplace("lights", std::make_unique<BufferDescriptorSet>(window, &this->descriptorSetLayouts.at("ssboF").handle, lightSSBODescriptorSettings));
	this->descriptorSets.emplace("lightMatrices", std::make_unique<BufferDescriptorSet>(window, &this->descriptorSetLayouts.at("ssboF").handle, lightMatricesSSBODescriptorSettings));
}

bool Renderer::checkSwapchain() {
	if (!this->recreateSwapchain)
		return false;

	// Handle minimisation
	GLFWwindow* glfwWindow = this->context.window->getGLFWwindow();

	int width, height;
	glfwGetFramebufferSize(glfwWindow, &width, &height);
	// Loop indefinitely until framebuffer size becomes non-zero (i.e. no longer minimised)
	while (width == 0 || height == 0) {
		glfwGetFramebufferSize(glfwWindow, &width, &height);
		glfwWaitEvents();
	}

	// Update shadow map resolutions
	this->sunShadowMapRes = this->shadowResolutions[this->sunShadowMapResIdx];
	this->pointShadowMapRes = this->shadowResolutions[this->pointShadowMapResIdx];

	VulkanWindow* window = this->context.window.get();

	// Wait for GPU to finish processing
	vkDeviceWaitIdle(window->getDevice()->getDevice());

	const SwapChanges swapChanges = window->getSwapchain()->recreate();

	if (swapChanges.changedFormat || this->forceRecreate)
		this->recreateFormatDependents();
	if (swapChanges.changedSize || this->forceRecreate)
		this->recreateSizeDependents();

	// Always recreate swapchain image view dependents
	this->recreateSwapViewDependents();

	// Mark all lights as dirty to re-render the shadow maps
	for (Light& light : *this->lights) {
		light.markDirty();
	}

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
	this->camera->update(this->context.window->getGLFWwindow(), timeDelta);

	this->uniforms.mvpUniform.projection = this->camera->getProjection();
	this->uniforms.mvpUniform.view = this->camera->getView();
	this->uniforms.mvpUniform.camPos = glm::vec4(this->camera->getPosition(), 1.0f);
	
	this->uniforms.cameraPlanesUniform._far = this->camera->getFarPlane();
	this->uniforms.cameraPlanesUniform._near = this->camera->getNearPlane();

	this->uniforms.projectiveUniform.projection = this->camera->getProjection();
	this->uniforms.projectiveUniform.invProjection = this->camera->getInvProjection();

	this->uniforms.inverseMatricesUniform.invViewProj = this->camera->getInvView() * this->camera->getInvProjection();
	this->uniforms.inverseMatricesUniform.invProj = this->camera->getInvProjection();
	this->uniforms.inverseMatricesUniform.invView = this->camera->getInvView();

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
			// Update light matrix
			this->ssbos.lightMatrices.at(directionalLightIndex) = this->sunMatrices.projection.get() * this->sunMatrices.view.get();

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

	if (!this->renderCameraFrustumBounds) return;

	std::array<glm::vec4, 8> frustumCornersArr = this->camera->getFrustumCorners();
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

		VkCommandBuffer uploadCmd = VkUtils::createCommandBuffer(*this->context.window, this->context.window->getDevice()->getCmdPool());

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

		VkCommandBuffer uploadCmd = VkUtils::createCommandBuffer(*this->context.window, this->context.window->getDevice()->getCmdPool());

		VkUtils::beginCommandBuffer(uploadCmd);

		BakedModelLoader::copyToGPU(uploadCmd, this->lineMeshData.posBuffer, this->lineMeshData.posBufferStaging, frustumCorners);
		BakedModelLoader::copyToGPU(uploadCmd, this->lineMeshData.colBuffer, this->lineMeshData.colBufferStaging, lineColours);
		BakedModelLoader::copyToGPU(uploadCmd, this->lineMeshData.indicesBuffer, this->lineMeshData.indicesBufferStaging, lineIndices);

		VkUtils::endAndSubmitCommandBuffer(*this->context.window, uploadCmd);
	}
}

void Renderer::render() {
	// Begin command buffer
	RendererUtils::bindCommandBuffer(this->cmdBuffers, this->frameIndex);
	RendererUtils::beginCommandBuffer(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
	this->driver->getTimestampManager().resetGPUQueryPool();
	this->driver->getTimestampManager().writeGPUTimestamp("entireFrame", VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);

	// Update uniform and shader storage buffers
	// TODO: add a dirty flag feature somewhere as these buffers do not need to be updated
	// every frame and can save on computation time
	RendererUtils::updateUniformBuffer(this->getUniformBuffer("mvp"));
	RendererUtils::updateUniformBuffer(this->getUniformBuffer("invMatrices"));
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

		if (this->vsmShadowsEnabled) {
			this->renderVSMShadowMaps();
		} else {
			this->renderShadowMaps();
		}
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
	if (this->numSpotLights == 0) {
		RendererUtils::imageBarrier(this->getTextureBuffer("spotArrayShadow")->getImage().image, 0, 0,
			VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
			VkImageSubresourceRange{ VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 });
	}

	// Scene pass
	if (this->renderingType) {
		this->renderDeferred();
	} else {
		this->renderForward();
	}

	// Post processing effects
	VkDescriptorSet readHDR = RendererUtils::getDescriptorSetHandle(this->getDescriptorSet("HDR"));
	VkDescriptorSet readLDR = RendererUtils::getDescriptorSetHandle(this->getDescriptorSet("LDR"));
	VkDescriptorSet writeHDR = RendererUtils::getDescriptorSetHandle(this->getDescriptorSet("intermediateHDR"));
	VkDescriptorSet writeLDR = RendererUtils::getDescriptorSetHandle(this->getDescriptorSet("intermediateLDR"));

	// HDR framebuffers
	WriteToTargetFramebuffer* framebufferHDR1 = dynamic_cast<WriteToTargetFramebuffer*>(this->getFramebuffer("writeToHDR"));
	WriteToTargetFramebuffer* framebufferHDR2 = dynamic_cast<WriteToTargetFramebuffer*>(this->getFramebuffer("writeToIntermediateHDR"));
	// LDR framebuffers
	WriteToTargetFramebuffer* framebufferLDR1 = dynamic_cast<WriteToTargetFramebuffer*>(this->getFramebuffer("writeToLDR"));
	WriteToTargetFramebuffer* framebufferLDR2 = dynamic_cast<WriteToTargetFramebuffer*>(this->getFramebuffer("writeToIntermediateLDR"));

	// Construct pairs to ping-pong
	VkDescriptorSetPair readImages{ readHDR, readLDR };
	VkDescriptorSetPair writeImages{ writeHDR, writeLDR };

	WriteToFramebufferPair framebuffers1{ framebufferHDR2, framebufferLDR2 }; // Intermediate HDR/LDR
	WriteToFramebufferPair framebuffers2{ framebufferHDR1, framebufferLDR1 }; // HDR/LDR

	// Swaps
	// 0: - read: hdr, ldr				write: intermediate_hdr/ldr
	// 1: - read: intermediate_hdr/ldr  write: hdr, ldr
	// 2: - read: hdr, ldr				write: intermediate_hdr/ldr
	// 3: - read: intermediate_hdr/ldr  write: hdr, ldr

	// Example scenarios to visualise the read and write targets
	// Scenario 1 - All PPE on
	// 0: bloom   - read: hdr				write: intermediate_hdr
	// 1: tonemap - read: intermediate_hdr  write: ldr
	// 2: fxaa    - read: ldr				write: intermediate_ldr
	// 3: moasic  - read: intermediate_ldr  write: ldr

	// Scenario 2 - No bloom
	// 0: tonemap - read: hdr				write: intermediate_ldr
	// 1: fxaa    - read: intermediate_ldr  write: ldr
	// 2: mosaic  - read: ldr				write: intermediate_ldr

	// Scenario 3 - Just mosaic (tonemapping is *always* on)
	// 0: tonemap - read: hdr				write: intermediate_ldr
	// 1: mosaic  - read: intermediate_ldr  write: ldr

	// Scenario 4 - Just tonemapping
	// 0: tonemap - read: hdr				write: intermediate_ldr

	// Need to keep track of the last written to texture buffer so we know
	// which to blit to the swapchain image
	TextureBuffer* lastWrittenToImage = this->getTextureBuffer("HDR");

	for (const auto& [effectName, effect] : this->postProcessingEffects) {
		if (!effect->getEnabled()) continue;

		lastWrittenToImage = effect->apply(framebuffers1, this->imageIndex, readImages);

		std::swap(readImages, writeImages);
		std::swap(framebuffers1, framebuffers2);
	}

	// Blit image to swapchain
	VkImage srcImage = lastWrittenToImage->getImage().image;
	VkImage swapchainImage = this->context.window->getSwapchain()->getImage(this->imageIndex);

	RendererUtils::blitImageToSwapchain(
		srcImage, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, swapchainImage,
		this->context.window->getSwapchain()->getExtent(), VK_FILTER_LINEAR);

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

	// Draw skybox
	// (a skybox render pass would look identical to postProcess so we just use that)
	RendererUtils::beginRenderPass(this->getRenderPass("postProcessHDR"), this->getFramebuffer("writeToHDR"), this->imageIndex);
	RendererUtils::bindGraphicPipeline(this->getPipeline("skybox")->getHandle());
	RendererUtils::bindGraphicDescriptorSets(
		this->getPipelineLayout("skybox")->getHandle(), 0, 1, &RendererUtils::getDescriptorSetHandle(this->getDescriptorSet("mvp")));
	RendererUtils::bindGraphicDescriptorSets(
		this->getPipelineLayout("skybox")->getHandle(), 1, 1, &RendererUtils::getDescriptorSetHandle(this->getDescriptorSet("skybox")));
	RendererUtils::drawDirect(36, 1, 0, 0);
	RendererUtils::endRenderPass();

	glsl::SunPC sunPC = {
		.sunDir = glm::vec4(this->ssbos.lights[this->sunLightIndex].directionAndMapIndex),
		.sunColour = glm::vec4(this->ssbos.lights[this->sunLightIndex].colourAndIntensity),
		.params = glm::vec4(this->sunUpperStep, this->sunLowerStep, this->sunIntensity, 0.0f)
	};

	// Draw Sun
	RendererUtils::beginRenderPass(this->getRenderPass("sun"), this->getFramebuffer("writeToHDR"), this->imageIndex);
	RendererUtils::bindGraphicPipeline(this->getPipeline("sun")->getHandle());
	RendererUtils::bindPushConstant(this->getPipelineLayout("sun")->getHandle(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(glsl::SunPC), &sunPC);
	RendererUtils::bindGraphicDescriptorSets(
		this->getPipelineLayout("sun")->getHandle(), 0, 1, &RendererUtils::getDescriptorSetHandle(this->getDescriptorSet("mvp")));
	RendererUtils::bindGraphicDescriptorSets(
		this->getPipelineLayout("sun")->getHandle(), 1, 1, &RendererUtils::getDescriptorSetHandle(this->getDescriptorSet("invMatrices")));
	RendererUtils::drawDirect(3, 1, 0, 0);
	RendererUtils::endRenderPass();

	// Do SSAO pass
	if (this->ssaoEnabled) {
		this->ssaoEffect->apply(this->imageIndex, true);
	} else {
		RendererUtils::imageBarrier(this->getTextureBuffer("ssaoVblur")->getImage().image, 0, 0,
			VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
			VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 });
	}

	this->driver->getTimestampManager().writeGPUTimestamp("forward", VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);

	// Forward pass
	RendererUtils::beginRenderPass(this->getRenderPass("forward"), this->getFramebuffer("forward"), this->imageIndex);
	RendererUtils::bindGraphicPipeline(this->getPipeline("forward")->getHandle());

	glsl::LightsAndEmissive lightsAndEmissive = {
		.numLights = this->numLights,
		.emissiveStrength = this->emissiveStrength,
		.brightnessThreshold = this->brightnessThreshold,
		.shadowBias = this->shadowBias,
		.ssaoEnabled = this->ssaoEnabled,
		.ssaoExp = this->ssaoExp
	};

	RendererUtils::bindPushConstant(
		this->getPipelineLayout("forward")->getHandle(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(glsl::LightsAndEmissive), &lightsAndEmissive);

	std::vector<VkDescriptorSet> descriptorSets;
	descriptorSets.reserve(8);
	// Add descriptors that will always be present in same order as in shader
	descriptorSets.emplace_back(RendererUtils::getDescriptorSetHandle(this->getDescriptorSet("mvp")));
	descriptorSets.emplace_back(RendererUtils::getDescriptorSetHandle(this->getDescriptorSet("lights")));
	descriptorSets.emplace_back(RendererUtils::getDescriptorSetHandle(this->getDescriptorSet("ssaoSampler")));

	if (this->shadowsEnabled) {
		descriptorSets.emplace_back(RendererUtils::getDescriptorSetHandle(this->getDescriptorSet("pointLightShadows")));
		descriptorSets.emplace_back(RendererUtils::getDescriptorSetHandle(this->getDescriptorSet("directionalLightShadow")));
		descriptorSets.emplace_back(RendererUtils::getDescriptorSetHandle(this->getDescriptorSet("spotLightShadows")));
		descriptorSets.emplace_back(RendererUtils::getDescriptorSetHandle(this->getDescriptorSet("cameraPlanes")));
		descriptorSets.emplace_back(RendererUtils::getDescriptorSetHandle(this->getDescriptorSet("lightMatrices")));
	}

	RendererUtils::bindGraphicDescriptorSets(
		this->getPipelineLayout("forward")->getHandle(), 0, descriptorSets.size(),
		descriptorSets.data());

	RendererUtils::setCullMode(VK_CULL_MODE_BACK_BIT);

	std::function<void(MeshData&)> perMeshCallback = [this, &descriptorSets](MeshData& meshData) {
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
	this->uniforms.mvpUniform.projection = this->sunMatrices.projection.get();
	this->uniforms.mvpUniform.view = this->sunMatrices.view.get();
	this->uniforms.mvpUniform.camPos = glm::vec4(glm::vec3(lightStruct.positionAndLightType), 1.0f);

	RendererUtils::updateUniformBuffer(this->getUniformBuffer("mvp"));

	// Sun position view
	RendererUtils::beginRenderPass(this->getRenderPass("sunView"), this->getFramebuffer("sunView"), this->imageIndex);
	RendererUtils::bindGraphicPipeline(this->getPipeline("sunView")->getHandle());

	descriptorSets.clear();
	descriptorSets.emplace_back(RendererUtils::getDescriptorSetHandle(this->getDescriptorSet("mvp")));

	RendererUtils::bindGraphicDescriptorSets(
		this->getPipelineLayout("sunView")->getHandle(), 0, descriptorSets.size(),
		descriptorSets.data());

	RendererUtils::setCullMode(VK_CULL_MODE_BACK_BIT);

	perMeshCallback = [this, &descriptorSets](MeshData& meshData) {
		// Material descriptor should always be last descriptor since its per mesh
		RendererUtils::bindGraphicDescriptorSets(
			this->getPipelineLayout("sunView")->getHandle(), descriptorSets.size(), 1,
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

	// Render frustum bounding box
	if (this->renderCameraFrustumBounds) {
		RendererUtils::bindGraphicPipeline(this->getPipeline("lineDebug")->getHandle());

		RendererUtils::bindGraphicDescriptorSets(
			this->getPipelineLayout("lineDebug")->getHandle(), 0, 1,
			&RendererUtils::getDescriptorSetHandle(this->getDescriptorSet("mvp")));

		RendererUtils::drawLineMesh(this->lineMeshData);
	}

	RendererUtils::endRenderPass();
}

void Renderer::renderDeferred() {
	std::vector<MeshData>& meshData = this->driver->getMeshData();

	this->driver->getTimestampManager().writeGPUTimestamp("deferredWriting", VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);

	// Writing to G-buffers pass
	RendererUtils::beginRenderPass(this->getRenderPass("deferredWriting"), this->getFramebuffer("deferredWriting"), this->imageIndex);
	RendererUtils::bindGraphicPipeline(this->getPipeline("deferredWriting")->getHandle());

	RendererUtils::bindGraphicDescriptorSets(
		this->getPipelineLayout("deferredWriting")->getHandle(), 0, 1,
		&RendererUtils::getDescriptorSetHandle(this->getDescriptorSet("mvp")));

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

	RendererUtils::endRenderPass();

	this->driver->getTimestampManager().writeGPUTimestamp("deferredWriting", VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

	// Draw skybox
	// (a skybox render pass would look identical to postProcess so we just use that)
	RendererUtils::beginRenderPass(this->getRenderPass("postProcessHDR"), this->getFramebuffer("writeToHDR"), this->imageIndex);
	RendererUtils::bindGraphicPipeline(this->getPipeline("skybox")->getHandle());
	RendererUtils::bindGraphicDescriptorSets(
		this->getPipelineLayout("skybox")->getHandle(), 0, 1, &this->getDescriptorSet("mvp")->getHandle());
	RendererUtils::bindGraphicDescriptorSets(
		this->getPipelineLayout("skybox")->getHandle(), 1, 1, &this->getDescriptorSet("skybox")->getHandle());
	RendererUtils::drawDirect(36, 1, 0, 0);
	RendererUtils::endRenderPass();

	glsl::SunPC sunPC = {
		.sunDir = glm::vec4(this->ssbos.lights[this->sunLightIndex].directionAndMapIndex),
		.sunColour = glm::vec4(this->ssbos.lights[this->sunLightIndex].colourAndIntensity),
		.params = glm::vec4(this->sunUpperStep, this->sunLowerStep, this->sunIntensity, 0.0f)
	};

	// Draw Sun
	RendererUtils::beginRenderPass(this->getRenderPass("sun"), this->getFramebuffer("writeToHDR"), this->imageIndex);
	RendererUtils::bindGraphicPipeline(this->getPipeline("sun")->getHandle());
	RendererUtils::bindPushConstant(this->getPipelineLayout("sun")->getHandle(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(glsl::SunPC), &sunPC);
	RendererUtils::bindGraphicDescriptorSets(this->getPipelineLayout("sun")->getHandle(), 0, 1, &this->getDescriptorSet("mvp")->getHandle());
	RendererUtils::bindGraphicDescriptorSets(this->getPipelineLayout("sun")->getHandle(), 1, 1, &this->getDescriptorSet("invMatrices")->getHandle());
	RendererUtils::drawDirect(3, 1, 0, 0);
	RendererUtils::endRenderPass();

	// SSAO Pass
	if (this->ssaoEnabled) {
		this->ssaoEffect->apply(this->imageIndex);
	} else {
		RendererUtils::imageBarrier(this->getTextureBuffer("ssaoVblur")->getImage().image, 0, 0,
			VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
			VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 });
	}

	this->driver->getTimestampManager().writeGPUTimestamp("deferredShading", VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);

	// Shading pass
	RendererUtils::beginRenderPass(this->getRenderPass("deferredShading"), this->getFramebuffer("deferredShading"), this->imageIndex);
	RendererUtils::bindGraphicPipeline(this->getPipeline("deferredShading")->getHandle());

	glsl::LightsAndEmissive lightsAndEmissive = {
		.numLights = this->numLights,
		.emissiveStrength = this->emissiveStrength,
		.brightnessThreshold = this->brightnessThreshold,
		.shadowBias = this->shadowBias,
		.ssaoEnabled = this->ssaoEnabled,
		.ssaoExp = this->ssaoExp
	};

	RendererUtils::bindPushConstant(
		this->getPipelineLayout("deferredShading")->getHandle(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(glsl::LightsAndEmissive), &lightsAndEmissive);

	std::vector<VkDescriptorSet> descriptorSets;
	descriptorSets.reserve(10);
	descriptorSets.emplace_back(RendererUtils::getDescriptorSetHandle(this->getDescriptorSet("deferredInputs")));
	descriptorSets.emplace_back(RendererUtils::getDescriptorSetHandle(this->getDescriptorSet("mvp")));
	descriptorSets.emplace_back(RendererUtils::getDescriptorSetHandle(this->getDescriptorSet("lights")));
	descriptorSets.emplace_back(RendererUtils::getDescriptorSetHandle(this->getDescriptorSet("invMatrices")));
	descriptorSets.emplace_back(RendererUtils::getDescriptorSetHandle(this->getDescriptorSet("ssaoSampler")));

	if (this->shadowsEnabled) {
		descriptorSets.emplace_back(RendererUtils::getDescriptorSetHandle(this->getDescriptorSet("pointLightShadows")));
		descriptorSets.emplace_back(RendererUtils::getDescriptorSetHandle(this->getDescriptorSet("directionalLightShadow")));
		descriptorSets.emplace_back(RendererUtils::getDescriptorSetHandle(this->getDescriptorSet("spotLightShadows")));
		descriptorSets.emplace_back(RendererUtils::getDescriptorSetHandle(this->getDescriptorSet("cameraPlanes")));
		descriptorSets.emplace_back(RendererUtils::getDescriptorSetHandle(this->getDescriptorSet("lightMatrices")));
	}

	RendererUtils::bindGraphicDescriptorSets(
		this->getPipelineLayout("deferredShading")->getHandle(), 0, descriptorSets.size(),
		descriptorSets.data());

	RendererUtils::drawDirect(3, 1, 0, 0);

	RendererUtils::endRenderPass();

	this->driver->getTimestampManager().writeGPUTimestamp("deferredShading", VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
}

void Renderer::renderDebugViews() {
	std::vector<MeshData>& meshData = this->driver->getMeshData();

	this->driver->getTimestampManager().writeGPUTimestamp("debugViews", VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);

	// Determine debug state flags
	bool isOvervisualisation = this->debugState > 6;

	if (isOvervisualisation) {
		// Set clear colour to a dark green to create a 'negative'-like image, but also restore
		// the original clear colour afterwards for when we disable overvisualisation
		VkClearValue originalValue = this->getRenderPass("debug")->getClearValues().at(0);
		this->getRenderPass("debug")->getClearValues().at(0) = { {0.0f, 0.3f, 0.0f, 1.0f} };
		RendererUtils::beginRenderPass(this->getRenderPass("debug"), this->getFramebuffer("debug"), this->imageIndex);
		this->getRenderPass("debug")->getClearValues().at(0) = originalValue;
	} else {
		RendererUtils::beginRenderPass(this->getRenderPass("debug"), this->getFramebuffer("debug"), this->imageIndex);
	}

	std::vector<VkDescriptorSet> descriptorSets;
	descriptorSets.reserve(3);
	descriptorSets.emplace_back(RendererUtils::getDescriptorSetHandle(this->getDescriptorSet("mvp")));

	if (!isOvervisualisation) {
		RendererUtils::bindGraphicPipeline(this->getPipeline("debugViews")->getHandle());

		debugStatePC debugState = {
			.lightCount = this->numLights,
			.debugState = this->debugState
		};

		RendererUtils::bindPushConstant(
			this->getPipelineLayout("debugViews")->getHandle(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(debugStatePC), &debugState);

		descriptorSets.emplace_back(RendererUtils::getDescriptorSetHandle(this->getDescriptorSet("lights")));
		descriptorSets.emplace_back(RendererUtils::getDescriptorSetHandle(this->getDescriptorSet("cameraPlanes")));

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
	VkImage srcImage = this->getTextureBuffer("HDR")->getImage().image;
	VkImage swapchainImage = this->context.window->getSwapchain()->getImage(this->imageIndex);

	RendererUtils::blitImageToSwapchain(
		srcImage, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, swapchainImage,
		this->context.window->getSwapchain()->getExtent(), VK_FILTER_LINEAR);

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
	std::uint32_t spotLightIndex = 0;

	// For each light
	for (std::size_t i = 0; i < this->lights->size(); i++) {
		Light& light = this->lights->at(i);

		// Check if we need to re-render this lights shadow map
		if (!light.isDirty()) continue;

		switch (light.getLightType()) {
		case LightType::POINT:
		{
			assert(this->numPointLights != 0 && "Trying to render a point light shadow map but numPointLights is 0?");

			constexpr glm::vec3 directions[6] = {
				glm::vec3(1.0f,  0.0f,  0.0f),
				glm::vec3(-1.0f,  0.0f,  0.0f),
				glm::vec3(0.0f,  1.0f,  0.0f),
				glm::vec3(0.0f, -1.0f,  0.0f),
				glm::vec3(0.0f,  0.0f,  1.0f),
				glm::vec3(0.0f,  0.0f, -1.0f),
			};

			constexpr glm::vec3 upVectors[6] = {
				glm::vec3(0.0f, -1.0f,  0.0f),
				glm::vec3(0.0f, -1.0f,  0.0f),
				glm::vec3(0.0f,  0.0f,  1.0f),
				glm::vec3(0.0f,  0.0f, -1.0f),
				glm::vec3(0.0f, -1.0f,  0.0f),
				glm::vec3(0.0f, -1.0f,  0.0f),
			};

			const glm::mat4 cubePerspective = glm::perspective(glm::radians(90.0f), 1.0f, this->camera->getNearPlane(), this->camera->getFarPlane());

			// Render to each face of the cube map
			for (std::size_t face = 0; face < 6; face++) {
				// Calculate layer index
				std::uint32_t layer = (pointLightIndex * 6) + face;

				RendererUtils::beginRenderPass(this->getRenderPass("shadow"), this->getFramebuffer("pointArrayShadows"), layer);

				RendererUtils::bindGraphicPipeline(this->getPipeline("cubemapShadow")->getHandle());

				glm::mat4 cubeView = glm::lookAt(light.getPosition(), light.getPosition() + directions[face], upVectors[face]);
				glm::mat4 cubeMatrix = cubePerspective * cubeView;

				glsl::CubemapPC fragPC = {
					.lightPos = glm::vec4(light.getPosition(), 1.0f),
					.farPlane = this->camera->getFarPlane()
				};

				RendererUtils::bindPushConstant(this->getPipelineLayout("cubemapShadow")->getHandle(),
					VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), &cubeMatrix);
				RendererUtils::bindPushConstant(this->getPipelineLayout("cubemapShadow")->getHandle(),
					VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(glm::mat4), sizeof(glsl::CubemapPC), &fragPC);

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

			RendererUtils::beginRenderPass(this->getRenderPass("varianceShadow"), this->getFramebuffer("directionalShadow"), directionalLightIndex);

			RendererUtils::bindGraphicPipeline(this->getPipeline("varianceShadow")->getHandle());

			glm::mat4 lightMatrix = this->ssbos.lightMatrices.at(directionalLightIndex);

			struct TwoMatPC {
				glm::mat4 lightViewProj{};
				glm::mat4 lightView{};
			};

			TwoMatPC twoMatPC = {
				.lightViewProj = this->sunMatrices.projection.get() * this->sunMatrices.view.get(),
				.lightView = this->sunMatrices.view.get()
			};

			RendererUtils::bindPushConstant(this->getPipelineLayout("varianceShadow")->getHandle(),
				VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(TwoMatPC), &twoMatPC);
#ifndef NDEBUG
			int projType = 1;

			//RendererUtils::bindPushConstant(this->getPipelineLayout("shadow")->getHandle(),
				//VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(glm::mat4), sizeof(int), &projType);
#endif
			RendererUtils::bindGraphicDescriptorSets(
				this->getPipelineLayout("varianceShadow")->getHandle(), 0, 1,
				&RendererUtils::getDescriptorSetHandle(this->getDescriptorSet("cameraPlanes")));

			//RendererUtils::setDepthBias(this->depthBiasConstant, 0.0f, this->depthBiasSlopeFactor);

			auto perMeshCallback = [this](MeshData& meshData) {
				RendererUtils::bindGraphicDescriptorSets(
					this->getPipelineLayout("varianceShadow")->getHandle(), 1, 1,
					&this->alphaMaskDescriptors.at(meshData.materialId));
				};

			for (std::size_t i = 0; i < meshData.size(); i++)
				RendererUtils::drawMeshGeometry(meshData[i], true, perMeshCallback);

			RendererUtils::endRenderPass();

			// Blur variance shadow map
			// Horizontal pass
			int direction = 0;
			RendererUtils::beginRenderPass(this->getRenderPass("varianceBlur"), this->getFramebuffer("writeToVarianceBlur"), directionalLightIndex);
			RendererUtils::bindGraphicPipeline(this->getPipeline("varianceBlur")->getHandle());
			RendererUtils::bindGraphicDescriptorSets(
				this->getPipelineLayout("bloom")->getHandle(), 0, 1, 
				&RendererUtils::getDescriptorSetHandle(this->getDescriptorSet("directionalLightShadow")));
			RendererUtils::bindPushConstant(this->getPipelineLayout("bloom")->getHandle(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(int), &direction);
			RendererUtils::drawDirect(3, 1, 0, 0);
			RendererUtils::endRenderPass();

			// Vertical pass
			direction = 1;
			RendererUtils::beginRenderPass(this->getRenderPass("varianceBlur"), this->getFramebuffer("writeToDirectionalShadow"), directionalLightIndex);
			RendererUtils::bindGraphicPipeline(this->getPipeline("varianceBlur")->getHandle());
			RendererUtils::bindGraphicDescriptorSets(
				this->getPipelineLayout("bloom")->getHandle(), 0, 1,
				&RendererUtils::getDescriptorSetHandle(this->getDescriptorSet("varianceBlur")));
			RendererUtils::bindPushConstant(this->getPipelineLayout("bloom")->getHandle(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(int), &direction);
			RendererUtils::drawDirect(3, 1, 0, 0);
			RendererUtils::endRenderPass();

			directionalLightIndex++;
			break;
		}
		case LightType::SPOT:
		{
			assert(this->numSpotLights != 0 && "Trying to render a spot light shadow map but numSpotLights is 0?");

			RendererUtils::beginRenderPass(this->getRenderPass("shadow"), this->getFramebuffer("spotArrayShadows"), spotLightIndex);

			RendererUtils::bindGraphicPipeline(this->getPipeline("shadowSpot")->getHandle());

			glm::mat4 lightMatrix = this->ssbos.lightMatrices.at((int)this->ssbos.lights[i].spotLightAndMatrixIndex.z);

			RendererUtils::bindPushConstant(this->getPipelineLayout("shadow")->getHandle(),
				VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), &lightMatrix);
#ifndef NDEBUG
			int projType = 0;

			RendererUtils::bindPushConstant(this->getPipelineLayout("shadow")->getHandle(),
				VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(glm::mat4), sizeof(int), &projType);

			RendererUtils::bindGraphicDescriptorSets(
				this->getPipelineLayout("shadow")->getHandle(), 0, 1,
				&RendererUtils::getDescriptorSetHandle(this->getDescriptorSet("cameraPlanes")));
#endif
			RendererUtils::setDepthBias(this->depthBiasConstant, 0.0f, this->depthBiasSlopeFactor);

			auto perMeshCallback = [this](MeshData& meshData) {
				RendererUtils::bindGraphicDescriptorSets(
					this->getPipelineLayout("shadow")->getHandle(), 1, 1,
					&this->alphaMaskDescriptors.at(meshData.materialId));
				};

			for (std::size_t i = 0; i < meshData.size(); i++)
				RendererUtils::drawMeshGeometry(meshData[i], true, perMeshCallback);

			RendererUtils::endRenderPass();

			spotLightIndex++;
			break;
		}
		}

		light.markClean();
	}

	this->driver->getTimestampManager().writeGPUTimestamp("shadows", VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
}

void Renderer::renderVSMShadowMaps() {
//	std::vector<MeshData>& meshData = this->driver->getMeshData();
//
//	this->driver->getTimestampManager().writeGPUTimestamp("vsmShadows", VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
//
//	// Some light-specific counters
//	std::uint32_t pointLightIndex = 0;
//	std::uint32_t directionalLightIndex = 0;
//	std::uint32_t spotLightIndex = 0;
//
//	// For each light
//	for (std::size_t i = 0; i < this->lights->size(); i++) {
//		Light& light = this->lights->at(i);
//
//		// Check if we need to re-render this lights shadow map
//		if (!light.isDirty()) continue;
//
//		switch (light.getLightType()) {
//		case LightType::POINT:
//		{
//			assert(this->numPointLights != 0 && "Trying to render a point light shadow map but numPointLights is 0?");
//
//			constexpr glm::vec3 directions[6] = {
//				glm::vec3(1.0f,  0.0f,  0.0f),
//				glm::vec3(-1.0f,  0.0f,  0.0f),
//				glm::vec3(0.0f,  1.0f,  0.0f),
//				glm::vec3(0.0f, -1.0f,  0.0f),
//				glm::vec3(0.0f,  0.0f,  1.0f),
//				glm::vec3(0.0f,  0.0f, -1.0f),
//			};
//
//			constexpr glm::vec3 upVectors[6] = {
//				glm::vec3(0.0f, -1.0f,  0.0f),
//				glm::vec3(0.0f, -1.0f,  0.0f),
//				glm::vec3(0.0f,  0.0f,  1.0f),
//				glm::vec3(0.0f,  0.0f, -1.0f),
//				glm::vec3(0.0f, -1.0f,  0.0f),
//				glm::vec3(0.0f, -1.0f,  0.0f),
//			};
//
//			const glm::mat4 cubePerspective = glm::perspective(glm::radians(90.0f), 1.0f, this->camera->getNearPlane(), this->camera->getFarPlane());
//
//			// Render to each face of the cube map
//			for (std::size_t face = 0; face < 6; face++) {
//				// Calculate layer index
//				std::uint32_t layer = (pointLightIndex * 6) + face;
//
//				RendererUtils::beginRenderPass(this->getRenderPass("varianceShadow"), this->getFramebuffer("pointArrayShadows"), layer);
//				RendererUtils::bindGraphicPipeline(this->getPipeline("varianceCubemapShadow")->getHandle());
//
//				glm::mat4 cubeView = glm::lookAt(light.getPosition(), light.getPosition() + directions[face], upVectors[face]);
//				glm::mat4 cubeMatrix = cubePerspective * cubeView;
//
//				glsl::CubemapPC fragPC = {
//					.lightPos = glm::vec4(light.getPosition(), 1.0f),
//					.farPlane = this->camera->getFarPlane()
//				};
//
//				RendererUtils::bindPushConstant(this->getPipelineLayout("cubemapShadow")->getHandle(),
//					VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), &cubeMatrix);
//				RendererUtils::bindPushConstant(this->getPipelineLayout("cubemapShadow")->getHandle(),
//					VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(glm::mat4), sizeof(glsl::CubemapPC), &fragPC);
//
//				for (std::size_t i = 0; i < meshData.size(); i++)
//					RendererUtils::drawMeshGeometry(meshData[i]);
//
//				RendererUtils::endRenderPass();
//			}
//
//			pointLightIndex++;
//			break;
//		}
//		case LightType::DIRECTIONAL:
//		{
//			assert(this->numDirectionalLights != 0 && "Trying to render a directional light shadow map but numDirectionalLights is 0?");
//
//			RendererUtils::beginRenderPass(this->getRenderPass("varianceShadow"), this->getFramebuffer("directionalShadow"), directionalLightIndex);
//
//			RendererUtils::bindGraphicPipeline(this->getPipeline("varianceShadow")->getHandle());
//
//			glm::mat4 lightMatrix = this->ssbos.lightMatrices.at(directionalLightIndex);
//
//			struct TwoMatPC {
//				glm::mat4 lightViewProj{};
//				glm::mat4 lightView{};
//			};
//
//			TwoMatPC twoMatPC = {
//				.lightViewProj = this->sunMatrices.projection.get() * this->sunMatrices.view.get(),
//				.lightView = this->sunMatrices.view.get()
//			};
//
//			RendererUtils::bindPushConstant(this->getPipelineLayout("varianceShadow")->getHandle(),
//				VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(TwoMatPC), &twoMatPC);
//			RendererUtils::bindGraphicDescriptorSets(
//				this->getPipelineLayout("varianceShadow")->getHandle(), 0, 1,
//				&RendererUtils::getDescriptorSetHandle(this->getDescriptorSet("cameraPlanes")));
//
//			auto perMeshCallback = [this](MeshData& meshData) {
//				RendererUtils::bindGraphicDescriptorSets(
//					this->getPipelineLayout("varianceShadow")->getHandle(), 1, 1,
//					&this->alphaMaskDescriptors.at(meshData.materialId));
//				};
//
//			for (std::size_t i = 0; i < meshData.size(); i++)
//				RendererUtils::drawMeshGeometry(meshData[i], true, perMeshCallback);
//
//			RendererUtils::endRenderPass();
//
//			// Blur variance shadow map
//			// Horizontal pass
//			int direction = 0;
//			RendererUtils::beginRenderPass(this->getRenderPass("varianceBlur"), this->getFramebuffer("writeToVarianceBlur"), directionalLightIndex);
//			RendererUtils::bindGraphicPipeline(this->getPipeline("varianceBlur")->getHandle());
//			RendererUtils::bindGraphicDescriptorSets(
//				this->getPipelineLayout("bloom")->getHandle(), 0, 1,
//				&RendererUtils::getDescriptorSetHandle(this->getDescriptorSet("directionalLightShadow")));
//			RendererUtils::bindPushConstant(this->getPipelineLayout("bloom")->getHandle(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(int), &direction);
//			RendererUtils::drawDirect(3, 1, 0, 0);
//			RendererUtils::endRenderPass();
//
//			// Vertical pass
//			direction = 1;
//			RendererUtils::beginRenderPass(this->getRenderPass("varianceBlur"), this->getFramebuffer("writeToDirectionalShadow"), directionalLightIndex);
//			RendererUtils::bindGraphicPipeline(this->getPipeline("varianceBlur")->getHandle());
//			RendererUtils::bindGraphicDescriptorSets(
//				this->getPipelineLayout("bloom")->getHandle(), 0, 1,
//				&RendererUtils::getDescriptorSetHandle(this->getDescriptorSet("varianceBlur")));
//			RendererUtils::bindPushConstant(this->getPipelineLayout("bloom")->getHandle(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(int), &direction);
//			RendererUtils::drawDirect(3, 1, 0, 0);
//			RendererUtils::endRenderPass();
//
//			directionalLightIndex++;
//			break;
//		}
//		case LightType::SPOT:
//		{
//			assert(this->numSpotLights != 0 && "Trying to render a spot light shadow map but numSpotLights is 0?");
//
//			RendererUtils::beginRenderPass(this->getRenderPass("shadow"), this->getFramebuffer("spotArrayShadows"), spotLightIndex);
//
//			RendererUtils::bindGraphicPipeline(this->getPipeline("shadowSpot")->getHandle());
//
//			glm::mat4 lightMatrix = this->ssbos.lightMatrices.at((int)this->ssbos.lights[i].spotLightAndMatrixIndex.z);
//
//			RendererUtils::bindPushConstant(this->getPipelineLayout("shadow")->getHandle(),
//				VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), &lightMatrix);
//#ifndef NDEBUG
//			int projType = 0;
//
//			RendererUtils::bindPushConstant(this->getPipelineLayout("shadow")->getHandle(),
//				VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(glm::mat4), sizeof(int), &projType);
//
//			RendererUtils::bindGraphicDescriptorSets(
//				this->getPipelineLayout("shadow")->getHandle(), 0, 1,
//				&RendererUtils::getDescriptorSetHandle(this->getDescriptorSet("cameraPlanes")));
//#endif
//			RendererUtils::setDepthBias(this->depthBiasConstant, 0.0f, this->depthBiasSlopeFactor);
//
//			auto perMeshCallback = [this](MeshData& meshData) {
//				RendererUtils::bindGraphicDescriptorSets(
//					this->getPipelineLayout("shadow")->getHandle(), 1, 1,
//					&this->alphaMaskDescriptors.at(meshData.materialId));
//				};
//
//			for (std::size_t i = 0; i < meshData.size(); i++)
//				RendererUtils::drawMeshGeometry(meshData[i], true, perMeshCallback);
//
//			RendererUtils::endRenderPass();
//
//			spotLightIndex++;
//			break;
//		}
//		}
//
//		light.markClean();
//	}
//
//	this->driver->getTimestampManager().writeGPUTimestamp("vsmShadows", VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
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
	vkDeviceWaitIdle(this->context.window->getDevice()->getDevice());
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

Camera* Renderer::getCamera() {
	return this->camera.get();
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

SSAOPreProcess* Renderer::getSSAOPreProcess() {
	return this->ssaoEffect.get();
}

std::vector<std::pair<std::string, _PostProcessingEffect>>& Renderer::getPostProcessingEffects() {
	return this->postProcessingEffects;
}

vk::Sampler& Renderer::getDefaultSampler() {
	return this->linearRepeatSampler;
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

LightMatrices& Renderer::getSunMatrices() {
	return this->sunMatrices;
}

std::uint32_t Renderer::getSunLightIndex() {
	return this->sunLightIndex;
}

void Renderer::setRecreateSwapchain(bool value, bool force) {
	this->recreateSwapchain = value;
	this->forceRecreate = force;
}
