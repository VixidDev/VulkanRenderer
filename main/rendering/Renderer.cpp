#include "Renderer.hpp"

#include <iostream>

#include "Error.hpp"
#include "toString.hpp"

#include "../Driver.hpp"
#include "../baked/BakedModel.hpp"
#include "../baked/BakedModelLoader.hpp"
#include "../imgui/imgui.h"
#include "../imgui/backends/imgui_impl_vulkan.h"
#include "../imgui/backends/imgui_impl_glfw.h"

#include "PipelineCreation.hpp"

#include "objects/impl/renderPasses/ForwardPass.hpp"
#include "objects/impl/renderPasses/ShadowPass.hpp"
#include "objects/impl/renderPasses/GUIPass.hpp"
#include "objects/impl/renderPasses/SunViewPass.hpp"

#include "objects/impl/pipelineLayouts/ForwardPipelineLayout.hpp"
#include "objects/impl/pipelineLayouts/ShadowPipelineLayout.hpp"
#include "objects/impl/pipelineLayouts/CubemapShadowPipelineLayout.hpp"
#include "objects/impl/pipelineLayouts/LineDebugPipelineLayout.hpp"

#include "objects/impl/pipelines/ForwardPipeline.hpp"
#include "objects/impl/pipelines/ShadowPipeline.hpp"
#include "objects/impl/pipelines/CubemapShadowPipeline.hpp"
#include "objects/impl/pipelines/LineDebugPipeline.hpp"

#include "objects/impl/textureBuffers/DepthTextureBuffer.hpp"
#include "objects/impl/textureBuffers/ShadowDepthTextureBuffer.hpp"
#include "objects/impl/textureBuffers/CubemapDepthTextureBuffer.hpp"
#include "objects/impl/textureBuffers/CubemapArrayDepthTextureBuffer.hpp"
#include "objects/impl/textureBuffers/ArrayColourTextureBuffer.hpp"
#include "objects/impl/textureBuffers/ArrayDepthTextureBuffer.hpp"
#include "objects/impl/textureBuffers/ColourTextureBuffer.hpp"

#include "objects/impl/framebuffers/ForwardFramebuffer.hpp"
#include "objects/impl/framebuffers/ShadowFramebuffer.hpp"
#include "objects/impl/framebuffers/CubemapFramebuffer.hpp"
#include "objects/impl/framebuffers/CubemapArrayFramebuffer.hpp"
#include "objects/impl/framebuffers/ArrayFramebuffer.hpp"
#include "objects/impl/framebuffers/GUIFramebuffer.hpp"
#include "objects/impl/framebuffers/SunFramebuffer.hpp"

#include "objects/impl/descriptorSets/BufferDescriptorSet.hpp"
#include "objects/impl/descriptorSets/ImageDescriptorSet.hpp"
#include "objects/impl/descriptorSets/ArrayImageDescriptorSet.hpp"

#include "../vulkan/VulkanDevice.hpp"
#include "../vulkan/VkUtils.hpp"
#include "RendererUtils.hpp"

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

Renderer::Renderer(Driver* driver) : driver(driver) {
	this->context.window = initialiseVulkanWindow();
	this->context.allocator = initialiseVulkanAllocator(*this->context.window);

	VulkanWindow* window = this->context.window.get();
	VulkanAllocator* allocator = this->context.allocator.get();

	this->camera = Camera(window, 45.0f, 0.01f, 128.0f, glm::vec3(-0.2972f, 7.3100f, -11.9532f), glm::vec3(0.0f, 0.0f, -1.0f));

	// Render passes
	this->renderPasses.emplace("forward", std::make_unique<ForwardPass>(window, &this->sampleCountSetting));
	this->renderPasses.emplace("shadow", std::make_unique<ShadowPass>(window, &this->sampleCountSetting));
	this->renderPasses.emplace("gui", std::make_unique<GUIPass>(window, &this->sampleCountSetting));
	this->renderPasses.emplace("sunView", std::make_unique<SunViewPass>(window, &this->sampleCountSetting));

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
		{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT } };

	this->descriptorSetLayouts.emplace("uboV", createDescriptorLayout(*window, uniformBufferV));
	this->descriptorSetLayouts.emplace("uboF", createDescriptorLayout(*window, uniformBufferF));
	this->descriptorSetLayouts.emplace("uboVF", createDescriptorLayout(*window, uniformBufferVF));
	this->descriptorSetLayouts.emplace("ssboF", createDescriptorLayout(*window, ssboF));
	this->descriptorSetLayouts.emplace("imageF", createDescriptorLayout(*window, imageF));
	this->descriptorSetLayouts.emplace("materials", createDescriptorLayout(*window, materialSettings));

	// Pipeline Layouts
	this->pipelineLayouts.emplace("forward", std::make_unique<ForwardPipelineLayout>(window, &this->descriptorSetLayouts, &this->shadowsEnabled));
	this->pipelineLayouts.emplace("shadow", std::make_unique<ShadowPipelineLayout>(window, &this->descriptorSetLayouts));
	this->pipelineLayouts.emplace("cubemapShadow", std::make_unique<CubemapShadowPipelineLayout>(window, &this->descriptorSetLayouts));
	this->pipelineLayouts.emplace("lineDebug", std::make_unique<LineDebugPipelineLayout>(window, &this->descriptorSetLayouts));

	// Pipelines
	this->pipelines.emplace("forward", std::make_unique<ForwardPipeline>(window, this->pipelineLayouts.at("forward").get(), this->renderPasses.at("forward").get(), &this->sampleCountSetting, &this->shadowsEnabled));
	this->pipelines.emplace("forwardSun", std::make_unique<ForwardPipeline>(window, this->pipelineLayouts.at("forward").get(), this->renderPasses.at("sunView").get(), &this->sampleCountSetting, &this->shadowsEnabled));
	this->pipelines.emplace("shadow", std::make_unique<ShadowPipeline>(window, this->pipelineLayouts.at("shadow").get(), this->renderPasses.at("shadow").get(), &this->sampleCountSetting, &this->shadowRes));
	this->pipelines.emplace("cubemapShadow", std::make_unique<CubemapShadowPipeline>(window, this->pipelineLayouts.at("cubemapShadow").get(), this->renderPasses.at("shadow").get(), &this->sampleCountSetting, &this->shadowRes));
	this->pipelines.emplace("lineDebug", std::make_unique<LineDebugPipeline>(window, this->pipelineLayouts.at("lineDebug").get(), this->renderPasses.at("sunView").get(), &this->sampleCountSetting));

	// Texture Buffers
	this->textureBuffers.emplace("depth", std::make_unique<DepthTextureBuffer>(&this->context, &this->sampleCountSetting));
	this->textureBuffers.emplace("sunView", std::make_unique<ColourTextureBuffer>(&this->context, &this->sampleCountSetting));

	// Framebuffers
	this->framebuffers.emplace("forward", std::make_unique<ForwardFramebuffer>(window, &this->textureBuffers, this->renderPasses.at("forward").get(), &this->sampleCountSetting));
	this->framebuffers.emplace("sun", std::make_unique<SunFramebuffer>(window, &this->textureBuffers, this->renderPasses.at("sunView").get(), &this->sampleCountSetting));
	this->framebuffers.emplace("gui", std::make_unique<GUIFramebuffer>(window, this->renderPasses.at("gui").get()));

	// Uniform Buffers
	VkPipelineStageFlags VFstageFlags = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	VkPipelineStageFlags VstageFlags = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT;
	VkPipelineStageFlags FstageFlags = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

	this->uniformBuffers.emplace("mvp", std::make_unique<UniformBuffer<glsl::MVPUniform>>(allocator, VFstageFlags, &this->uniforms.mvpUniform));
	this->uniformBuffers.emplace("depthMVP", std::make_unique<UniformBuffer<glsl::DepthMVPUniform>>(allocator, VstageFlags, &this->uniforms.depthMVPUniform));
	this->uniformBuffers.emplace("cameraPlanes", std::make_unique<UniformBuffer<glsl::CameraPlanesUniform>>(allocator, FstageFlags, &this->uniforms.cameraPlanesUniform));

	// Synchronisation
	for (std::size_t i = 0; i < window->swapViews.size(); i++) {
		this->cmdBuffers.emplace_back(createCommandBuffer(*window));
		this->frameDoneFences.emplace_back(createFence(*window, VK_FENCE_CREATE_SIGNALED_BIT));
		this->imageAvailableSemaphores.emplace_back(createSemaphore(*window));
		this->renderFinishedSemaphores.emplace_back(createSemaphore(*window));
	}

	// Samplers
	SamplerInfo defaultSamplerInfo = {
		VK_FILTER_LINEAR,
		VK_FILTER_LINEAR,
		VK_SAMPLER_ADDRESS_MODE_REPEAT,
		VK_SAMPLER_ADDRESS_MODE_REPEAT,
		VK_SAMPLER_ADDRESS_MODE_REPEAT };
	this->defaultSampler = createTextureSampler(*window, defaultSamplerInfo);
	SamplerInfo shadowMapSamplerInfo = {
		VK_FILTER_LINEAR,
		VK_FILTER_LINEAR,
		VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		1, VK_COMPARE_OP_LESS_OR_EQUAL };
	this->shadowMapSampler = createTextureSampler(*window, shadowMapSamplerInfo);

	// Descriptor Sets
	std::vector<DescriptorBufferSetting> mvpDescriptorSettings = {
		{ this->uniformBuffers.at("mvp")->getHandle(), VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER }};
	std::vector<DescriptorBufferSetting> depthDescriptorSettings = {
		{ this->uniformBuffers.at("depthMVP")->getHandle(), VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER}};
	std::vector<DescriptorBufferSetting> cameraPlanesDescriptorSettings = {
		{ this->uniformBuffers.at("cameraPlanes")->getHandle(), VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER }};
	std::vector<DescriptorImageSetting> sunViewDescriptorSettings = {
		{ this->textureBuffers.at("sunView").get(), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, this->defaultSampler.handle}};

	this->descriptorSets.emplace("mvp", std::make_unique<BufferDescriptorSet>(window, &this->descriptorSetLayouts.at("uboVF").handle, mvpDescriptorSettings));
	this->descriptorSets.emplace("depthMVP", std::make_unique<BufferDescriptorSet>(window, &this->descriptorSetLayouts.at("uboV").handle, depthDescriptorSettings));
	this->descriptorSets.emplace("cameraPlanes", std::make_unique<BufferDescriptorSet>(window, &this->descriptorSetLayouts.at("uboF").handle, cameraPlanesDescriptorSettings));
	this->descriptorSets.emplace("sunView", std::make_unique<ImageDescriptorSet>(window, &this->descriptorSetLayouts.at("imageF").handle, sunViewDescriptorSettings));
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
	this->textureBuffers.emplace("pointArrayShadows", std::make_unique<CubemapArrayDepthTextureBuffer>(&this->context, this->numPointLights, &this->shadowRes));
	this->textureBuffers.emplace("directionalArrayShadows", std::make_unique<ArrayDepthTextureBuffer>(&this->context, this->numDirectionalLights, &this->shadowRes));
#ifndef NDEBUG
	this->textureBuffers.emplace("pointArrayShadowsDebug", std::make_unique<ArrayColourTextureBuffer>(&this->context, this->numPointLights * 6, &this->shadowRes));
	this->textureBuffers.emplace("directionalArrayShadowsDebug", std::make_unique<ArrayColourTextureBuffer>(&this->context, this->numDirectionalLights, &this->shadowRes));
#endif

	if (this->numPointLights != 0) {
		std::initializer_list<TextureBuffer*> pointShadowTextures = {
			this->textureBuffers.at("pointArrayShadows").get(),
#ifndef NDEBUG
			this->textureBuffers.at("pointArrayShadowsDebug").get()
#endif
		};
		
		this->framebuffers.emplace("pointArrayShadows", std::make_unique<ArrayFramebuffer>(window, pointShadowTextures, this->renderPasses.at("shadow").get(), this->numPointLights * 6, &this->shadowRes));
	}

	if (this->numDirectionalLights != 0) {
		std::initializer_list<TextureBuffer*> directionalShadowTextures = {
			this->textureBuffers.at("directionalArrayShadows").get(),
#ifndef NDEBUG
			this->textureBuffers.at("directionalArrayShadowsDebug").get()
#endif
		};

		this->framebuffers.emplace("directionalArrayShadows", std::make_unique<ArrayFramebuffer>(window, directionalShadowTextures, this->renderPasses.at("shadow").get(), numDirectionalLights, &this->shadowRes));
	}
	
	// Light type counters (surely I can make a better system than this)
	int pointLightIndex = 0;
	int directionalLightIndex = 0;
	int spotLightIndex = 0;

	//glm::mat4 lightOrtho = glm::ortho(-30.0f, 30.0f, 30.0f, -30.0f, 0.01f, 1024.0f);

	// Populate ssbos
	for (std::size_t i = 0; i < lights->size(); i++) {
		Light light = lights->at(i);

		glsl::Light lightStruct = {
			.position = light.getPosition(),
			.direction = light.getDirection(),
			.colour = light.getColour(),
			.metadata = glm::ivec3(static_cast<int>(light.getLightType()), 0, 100)
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
			lightStruct.metadata.y = directionalLightIndex; // Shadow map index
			directionalLightIndex++;

			lightStruct.metadata.z = 1000; // Intensity

			LightMatrices lightMatrices = this->getLightMatricesForCameraFrustum(lightStruct);

			this->ssbos.lightMatrices.emplace_back(lightMatrices.projection * lightMatrices.view);
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
		{ this->textureBuffers.at("pointArrayShadows").get(), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, this->shadowMapSampler.handle } };
	std::vector<DescriptorImageSetting> directionalLightShadowsDescriptorSettings = {
		{ this->textureBuffers.at("directionalArrayShadows").get(), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, this->shadowMapSampler.handle } };

	this->descriptorSets.emplace("pointLightShadows", std::make_unique<ImageDescriptorSet>(window, &this->descriptorSetLayouts.at("imageF").handle, pointLightShadowsDescriptorSettings));
	this->descriptorSets.emplace("directionalLightShadows", std::make_unique<ImageDescriptorSet>(window, &this->descriptorSetLayouts.at("imageF").handle, directionalLightShadowsDescriptorSettings));

#ifndef NDEBUG
	std::vector<DescriptorImageSetting> pointLightShadowsDebugDescriptorSettings = {
		{ this->textureBuffers.at("pointArrayShadowsDebug").get(), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, this->shadowMapSampler.handle } };
	std::vector<DescriptorImageSetting> directionalLightShadowsDebugDescriptorSettings = {
		{ this->textureBuffers.at("directionalArrayShadowsDebug").get(), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, this->shadowMapSampler.handle } };

	this->descriptorSets.emplace("pointLightShadowsDebug", std::make_unique<ArrayImageDescriptorSet>(window, &this->descriptorSetLayouts.at("imageF").handle, pointLightShadowsDebugDescriptorSettings));
	this->descriptorSets.emplace("directionalLightShadowsDebug", std::make_unique<ArrayImageDescriptorSet>(window, &this->descriptorSetLayouts.at("imageF").handle, directionalLightShadowsDebugDescriptorSettings));
#endif

	std::vector<DescriptorBufferSetting> lightSSBODescriptorSettings = {
		{ this->shaderStorageBuffers.at("lights")->getHandle(), VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, this->shaderStorageBuffers.at("lights")->getBufferSize() }};

	std::vector<DescriptorBufferSetting> lightMatricesSSBODescriptorSettings = {
		{ this->shaderStorageBuffers.at("lightMatrices")->getHandle(), VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, this->shaderStorageBuffers.at("lightMatrices")->getBufferSize() }};

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

	this->recreateSwapchain = false;
	this->forceRecreate = false;

	return true;
}

bool Renderer::acquireSwapchainImage() {
	this->frameIndex++;
	this->frameIndex %= this->cmdBuffers.size();

	waitForFences(*this->context.window, this->frameDoneFences, this->frameIndex);

	if (const VkResult res = acquireNextSwapchainImage(*this->context.window, this->imageAvailableSemaphores, this->frameIndex, this->imageIndex); 
		res == VK_SUBOPTIMAL_KHR || res == VK_ERROR_OUT_OF_DATE_KHR) {
		this->recreateSwapchain = true;

		// If vkAcquireNextImageKHR returned VK_SUBOPTIMAL_KHR we can still render the frame and
		// recreate the swapchain before the next frame, this way the signalled semaphore from
		// vkAcquireNextImageKHR will be waited on during this frame's submission.
		if (res == VK_SUBOPTIMAL_KHR) {
			resetFences(*this->context.window, this->frameDoneFences, this->frameIndex);

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

	resetFences(*this->context.window, this->frameDoneFences, this->frameIndex);

	return false;
}

void Renderer::update(float timeDelta) {
	this->camera.update(this->context.window->window, timeDelta);

	this->uniforms.mvpUniform.projection = this->camera.getProjectionMat();
	this->uniforms.mvpUniform.view = this->camera.getViewMat();
	this->uniforms.mvpUniform.camPos = glm::vec4(this->camera.getPosition(), 1.0f);

	//glm::mat4 depthProjection = glm::ortho(-10.0f, 10.0f, 10.0f, -10.0f, 0.1f, 1000.0f);
	//glm::mat4 depthProjection = glm::perspective(glm::radians(90.0f), 1.0f, 0.01f, 256.0f);
	//depthProjection[1][1] *= -1.0f;
	//glm::mat4 depthView = glm::lookAt(
	//	glm::vec3(-0.2972f, 7.3100f, -11.9532f),
	//	glm::vec3(0.0f, 0.0f, -48.0f),
	//	glm::vec3(0.0f, 1.0f, 0.0f));

	//this->uniforms.depthMVPUniform.depthMVP = depthProjection * depthView;
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
			LightMatrices lightMatrices = this->getLightMatricesForCameraFrustum(lightStruct);

			// Update light matrix
			this->ssbos.lightMatrices.at(directionalLightIndex) = lightMatrices.projection * lightMatrices.view;

			directionalLightIndex++;
			break;
		}
		case LightType::SPOT:
		{
			break;
		}
		}
	}

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

		VkCommandBuffer uploadCmd = createCommandBuffer(*this->context.window);

		beginCommandBuffer(uploadCmd);

		BakedModelLoader::copyToGPU(uploadCmd, posLineGPU, posStaging, frustumCorners);
		BakedModelLoader::copyToGPU(uploadCmd, colLineGPU, colStaging, lineColours);
		BakedModelLoader::copyToGPU(uploadCmd, indexLineGPU, indexStaging, lineIndices);

		endAndSubmitCommandBuffer(*this->context.window, uploadCmd);

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

		VkCommandBuffer uploadCmd = createCommandBuffer(*this->context.window);

		beginCommandBuffer(uploadCmd);

		BakedModelLoader::copyToGPU(uploadCmd, this->lineMeshData.posBuffer, this->lineMeshData.posBufferStaging, frustumCorners);
		BakedModelLoader::copyToGPU(uploadCmd, this->lineMeshData.colBuffer, this->lineMeshData.colBufferStaging, lineColours);
		BakedModelLoader::copyToGPU(uploadCmd, this->lineMeshData.indicesBuffer, this->lineMeshData.indicesBufferStaging, lineIndices);

		endAndSubmitCommandBuffer(*this->context.window, uploadCmd);
	}
}

void Renderer::render() {
	// Begin command buffer
	this->cmdBuff = this->cmdBuffers[this->frameIndex];
	beginCommandBuffer(this->cmdBuff, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

	std::vector<MeshData>& meshData = this->driver->getMeshData();

	// Update uniform and shader storage buffers
	this->uniformBuffers.at("mvp")->update(this->cmdBuff);
	this->shaderStorageBuffers.at("lights")->update(this->cmdBuff);
	this->shaderStorageBuffers.at("lightMatrices")->update(this->cmdBuff);

	// Shadow pass
	if (this->shadowsEnabled) {
		this->uniformBuffers.at("cameraPlanes")->update(this->cmdBuff);

		this->renderShadowMaps(meshData);
	}

	// Transition any dummy light textures to respective layout
	if (this->numPointLights == 0) {
		Utils::imageBarrier(this->cmdBuff,
			this->textureBuffers.at("pointArrayShadows")->getImage().image,
			0, 0,
			VK_IMAGE_LAYOUT_UNDEFINED,
			VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VkImageSubresourceRange{ VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 6});
	}

	// Forward pass
	RendererUtils::beginRenderPass(this->cmdBuff, this->renderPasses.at("forward").get(), this->framebuffers.at("forward").get(), this->imageIndex);

	vkCmdBindPipeline(this->cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, this->pipelines.at("forward")->getHandle());
	vkCmdBindDescriptorSets(
		this->cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS,
		this->pipelineLayouts.at("forward")->getHandle(), 0, 1, 
		&this->descriptorSets.at("mvp")->getHandle(), 0, nullptr);

	if (this->shadowsEnabled) {
		vkCmdBindDescriptorSets(this->cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS,
			this->pipelineLayouts.at("forward")->getHandle(), 2, 1,
			&this->descriptorSets.at("pointLightShadows")->getHandle(), 0, nullptr);
		vkCmdBindDescriptorSets(this->cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS,
			this->pipelineLayouts.at("forward")->getHandle(), 3, 1,
			&this->descriptorSets.at("directionalLightShadows")->getHandle(), 0, nullptr);
		vkCmdBindDescriptorSets(this->cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS,
			this->pipelineLayouts.at("forward")->getHandle(), 4, 1,
			&this->descriptorSets.at("cameraPlanes")->getHandle(), 0, nullptr);
		vkCmdBindDescriptorSets(this->cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS,
			this->pipelineLayouts.at("forward")->getHandle(), 5, 1,
			&this->descriptorSets.at("lights")->getHandle(), 0, nullptr);
		vkCmdBindDescriptorSets(this->cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS,
			this->pipelineLayouts.at("forward")->getHandle(), 6, 1,
			&this->descriptorSets.at("lightMatrices")->getHandle(), 0, nullptr);
		vkCmdPushConstants(this->cmdBuff, this->pipelineLayouts.at("forward")->getHandle(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(int), &this->numLights);
	}

	vkCmdSetCullMode(this->cmdBuff, VK_CULL_MODE_BACK_BIT);

	auto perMeshCallback = [this](MeshData& meshData) {
		vkCmdBindDescriptorSets(
			this->cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS,
			this->pipelineLayouts.at("forward")->getHandle(), 1, 1,
			&this->driver->getMaterialDescriptors().at(meshData.materialId), 0, nullptr);
	};

	// Draw non-alpha masked meshes
	for (std::size_t i = 0; i < meshData.size(); i++) {
		if (meshData[i].hasAlphaMask) continue;

		this->drawMesh(meshData[i], perMeshCallback);
	}

	vkCmdSetCullMode(this->cmdBuff, VK_CULL_MODE_NONE);

	// Draw alpha masked meshes
	for (std::uint32_t i = 0; i < meshData.size(); i++) {
		if (!meshData[i].hasAlphaMask) continue;

		this->drawMesh(meshData[i], perMeshCallback);
	}

	RendererUtils::endRenderPass(this->cmdBuff);

	// This is stupid extra processing, we already process the matrices this frame
	// ...but im lazy (i should really cache stuff)
	for (std::size_t i = 0; i < this->lights->size(); i++) {
		Light light = this->lights->at(i);

		if (light.getLightType() == LightType::DIRECTIONAL) {
			glsl::Light lightStruct = this->ssbos.lights.at(i);

			LightMatrices lightMatrices = this->getLightMatricesForCameraFrustum(lightStruct);
			this->uniforms.mvpUniform.projection = lightMatrices.projection;
			this->uniforms.mvpUniform.view = lightMatrices.view;
			this->uniforms.mvpUniform.camPos = glm::vec4(lightStruct.position, 1.0f);
		}
	}

	this->uniformBuffers.at("mvp")->update(this->cmdBuff);
	
	// Sun position view
	RendererUtils::beginRenderPass(this->cmdBuff, this->renderPasses.at("sunView").get(), this->framebuffers.at("sun").get(), this->imageIndex);

	// Render scene
	vkCmdBindPipeline(this->cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, this->pipelines.at("forwardSun")->getHandle());
	vkCmdBindDescriptorSets(
		this->cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS,
		this->pipelineLayouts.at("forward")->getHandle(), 0, 1,
		&this->descriptorSets.at("mvp")->getHandle(), 0, nullptr);

	if (this->shadowsEnabled) {
		vkCmdBindDescriptorSets(this->cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS,
			this->pipelineLayouts.at("forward")->getHandle(), 2, 1,
			&this->descriptorSets.at("pointLightShadows")->getHandle(), 0, nullptr);
		vkCmdBindDescriptorSets(this->cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS,
			this->pipelineLayouts.at("forward")->getHandle(), 3, 1,
			&this->descriptorSets.at("directionalLightShadows")->getHandle(), 0, nullptr);
		vkCmdBindDescriptorSets(this->cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS,
			this->pipelineLayouts.at("forward")->getHandle(), 4, 1,
			&this->descriptorSets.at("cameraPlanes")->getHandle(), 0, nullptr);
		vkCmdBindDescriptorSets(this->cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS,
			this->pipelineLayouts.at("forward")->getHandle(), 5, 1,
			&this->descriptorSets.at("lights")->getHandle(), 0, nullptr);
		vkCmdBindDescriptorSets(this->cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS,
			this->pipelineLayouts.at("forward")->getHandle(), 6, 1,
			&this->descriptorSets.at("lightMatrices")->getHandle(), 0, nullptr);
		vkCmdPushConstants(this->cmdBuff, this->pipelineLayouts.at("forward")->getHandle(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(int), &this->numLights);
	}

	vkCmdSetCullMode(this->cmdBuff, VK_CULL_MODE_BACK_BIT);

	auto perMeshCallback2 = [this](MeshData& meshData) {
		vkCmdBindDescriptorSets(
			this->cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS,
			this->pipelineLayouts.at("forward")->getHandle(), 1, 1,
			&this->driver->getMaterialDescriptors().at(meshData.materialId), 0, nullptr);
		};

	// Draw non-alpha masked meshes
	for (std::size_t i = 0; i < meshData.size(); i++) {
		if (meshData[i].hasAlphaMask) continue;

		this->drawMesh(meshData[i], perMeshCallback2);
	}

	vkCmdSetCullMode(this->cmdBuff, VK_CULL_MODE_NONE);

	// Draw alpha masked meshes
	for (std::uint32_t i = 0; i < meshData.size(); i++) {
		if (!meshData[i].hasAlphaMask) continue;

		this->drawMesh(meshData[i], perMeshCallback2);
	}

	// Render frustum bounding box
	if (this->renderCameraFrustumBounds) {
		vkCmdBindPipeline(this->cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, this->pipelines.at("lineDebug")->getHandle());

		vkCmdBindDescriptorSets(
			this->cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS,
			this->pipelineLayouts.at("lineDebug")->getHandle(), 0, 1,
			&this->descriptorSets.at("mvp")->getHandle(), 0, nullptr);

		this->drawLineMesh(this->lineMeshData);
	}

	RendererUtils::endRenderPass(this->cmdBuff);

	RendererUtils::beginRenderPass(this->cmdBuff, this->renderPasses.at("gui").get(), this->framebuffers.at("gui").get(), this->imageIndex);

	if (ImGui::GetDrawData() != nullptr)
		ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), this->cmdBuff);

	RendererUtils::endRenderPass(this->cmdBuff);

	endCommandBuffer(*this->context.window, this->cmdBuff);
}

void Renderer::renderShadowMaps(std::vector<MeshData>& meshData) {
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
			
			const glm::mat4 cubePerspective = glm::perspective(glm::radians(90.0f), 1.0f, 0.01f, 128.0f);

			// Render to each face of the cube map
			for (std::size_t face = 0; face < 6; face++) {
				// Calculate layer index
				std::uint32_t layer = (pointLightIndex * 6) + face;

				RendererUtils::beginRenderPass(this->cmdBuff, this->renderPasses.at("shadow").get(), this->framebuffers.at("pointArrayShadows").get(), layer);

				vkCmdBindPipeline(this->cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, this->pipelines.at("cubemapShadow")->getHandle());
				
				glm::mat4 cubeView = glm::lookAt(light.getPosition(), light.getPosition() + directions[face], upVectors[face]);
				glm::mat4 cubeMatrix = cubePerspective * cubeView;

				struct cubemapFragmentPC {
					glm::vec4 lightPos;
					float farPlane;
				};

				cubemapFragmentPC fragPC = {
					.lightPos = glm::vec4(light.getPosition(), 1.0f),
					.farPlane = 128.0f
				};

				vkCmdPushConstants(this->cmdBuff, this->pipelineLayouts.at("cubemapShadow")->getHandle(), 
					VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), &cubeMatrix);
				vkCmdPushConstants(this->cmdBuff, this->pipelineLayouts.at("cubemapShadow")->getHandle(), 
					VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(glm::mat4), sizeof(glm::vec4) + sizeof(float), &fragPC);

				vkCmdSetDepthBias(this->cmdBuff, this->depthBiasConstant, 0.0f, this->depthBiasSlopeFactor);

				for (std::size_t i = 0; i < meshData.size(); i++)
					this->drawMeshGeometry(meshData[i]);

				RendererUtils::endRenderPass(this->cmdBuff);
			}

			pointLightIndex++;
			break;
		}
		case LightType::DIRECTIONAL:
		{
			assert(this->numDirectionalLights != 0 && "Trying to render a directional light shadow map but numDirectionalLights is 0?");

			RendererUtils::beginRenderPass(this->cmdBuff, this->renderPasses.at("shadow").get(), this->framebuffers.at("directionalArrayShadows").get(), directionalLightIndex);

			vkCmdBindPipeline(this->cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, this->pipelines.at("shadow")->getHandle());

			glm::mat4 lightMatrix = this->ssbos.lightMatrices.at(directionalLightIndex);

			vkCmdPushConstants(this->cmdBuff, this->pipelineLayouts.at("shadow")->getHandle(),
				VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), &lightMatrix);
#if !defined(NDEBUG)
			int projType = 1;

			vkCmdPushConstants(this->cmdBuff, this->pipelineLayouts.at("shadow")->getHandle(),
				VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(glm::mat4), sizeof(int), &projType);

			vkCmdBindDescriptorSets(this->cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS,
				this->pipelineLayouts.at("shadow")->getHandle(), 0, 1,
				&this->descriptorSets.at("cameraPlanes")->getHandle(), 0, nullptr);
#endif
			vkCmdSetDepthBias(this->cmdBuff, this->depthBiasConstant, 0.0f, this->depthBiasSlopeFactor);

			for (std::size_t i = 0; i < meshData.size(); i++)
				this->drawMeshGeometry(meshData[i]);

			RendererUtils::endRenderPass(this->cmdBuff);

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
}

void Renderer::drawMesh(MeshData& meshData, const std::function<void(MeshData&)>& perMeshCallback) {
	if (perMeshCallback)
		perMeshCallback(meshData);
	
	VkBuffer vBuffers[3] = { 
		meshData.posBuffer.buffer,
		meshData.texCoordBuffer.buffer,
		meshData.tbnFrameBuffer.buffer
	};
	VkBuffer iBuffer = meshData.indicesBuffer.buffer;
	VkDeviceSize vOffsets[3]{};
	VkDeviceSize iOffset{};

	vkCmdBindVertexBuffers(this->cmdBuff, 0, 3, vBuffers, vOffsets);
	vkCmdBindIndexBuffer(this->cmdBuff, iBuffer, iOffset, VK_INDEX_TYPE_UINT32);

	vkCmdDrawIndexed(this->cmdBuff, static_cast<std::uint32_t>(meshData.indicesCount), 1, 0, 0, 0);
}

void Renderer::drawMeshGeometry(MeshData& meshData, const std::function<void(MeshData&)>& perMeshCallback) {
	if (perMeshCallback)
		perMeshCallback(meshData);

	VkBuffer vBuffer = meshData.posBuffer.buffer;
	VkBuffer iBuffer = meshData.indicesBuffer.buffer;
	VkDeviceSize vOffset{};
	VkDeviceSize iOffset{};

	vkCmdBindVertexBuffers(this->cmdBuff, 0, 1, &vBuffer, &vOffset);
	vkCmdBindIndexBuffer(this->cmdBuff, iBuffer, iOffset, VK_INDEX_TYPE_UINT32);

	vkCmdDrawIndexed(this->cmdBuff, static_cast<std::uint32_t>(meshData.indicesCount), 1, 0, 0, 0);
}

void Renderer::drawLineMesh(LineMeshData& lineMeshData) {
	VkBuffer vBuffers[2] = {
		lineMeshData.posBuffer.buffer,
		lineMeshData.colBuffer.buffer,
	};
	VkBuffer iBuffer = lineMeshData.indicesBuffer.buffer;
	VkDeviceSize vOffsets[2]{};
	VkDeviceSize iOffset{};

	vkCmdBindVertexBuffers(this->cmdBuff, 0, 2, vBuffers, vOffsets);
	vkCmdBindIndexBuffer(this->cmdBuff, iBuffer, iOffset, VK_INDEX_TYPE_UINT32);

	vkCmdDrawIndexed(this->cmdBuff, static_cast<std::uint32_t>(lineMeshData.indicesCount), 1, 0, 0, 0);
}

LightMatrices Renderer::getLightMatricesForCameraFrustum(glsl::Light& lightStruct) {
	std::array<glm::vec4, 8> frustumCorners = this->camera.getFrustumCorners();
	
	// Get frustum center
	glm::vec3 frustumCenter(0.0f);
	for (const glm::vec3& corner : frustumCorners)
		frustumCenter += corner;
	frustumCenter /= static_cast<float>(frustumCorners.size());

	// Calc light pos
	lightStruct.position = frustumCenter - lightStruct.direction * 10.0f;

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
	float zMult = 10.0f;
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
}

void Renderer::finishRendering() {
	vkDeviceWaitIdle(this->context.window->device->device);

	ImGui_ImplVulkan_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
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

	// Recreate pipelines
	for (auto& pipeline : this->pipelines)
		pipeline.second->recreate();

	// Recreate framebuffers
	for (auto& framebuffer : this->framebuffers)
		framebuffer.second->recreate();
}

VulkanContext& Renderer::getContext() {
	return this->context;
}

Camera& Renderer::getCamera() {
	return this->camera;
}

VkRenderPass Renderer::getRenderPassHandle(const std::string& renderPass) {
	return this->renderPasses.at(renderPass).get()->getRenderPassHandle();
}

std::map<std::string, vk::DescriptorSetLayout>& Renderer::getDescriptorSetLayouts() {
	return this->descriptorSetLayouts;
}

TextureBuffer* Renderer::getTextureBuffer(const std::string& textureBuffer) {
	TextureBuffer* ret = nullptr;
	
	try {
		ret = this->textureBuffers.at(textureBuffer).get();
	} catch (const std::out_of_range&) {
		std::printf("Could not find: %s in 'textureBuffers'\n", textureBuffer.c_str());
	}

	return ret;
}

DescriptorSet* Renderer::getDescriptorSet(const std::string& descriptorSet) {
	DescriptorSet* ret = nullptr;

	try {
		ret = this->descriptorSets.at(descriptorSet).get();
	} catch (const std::out_of_range&) {
		std::printf("Could not find: %s in 'descriptorSets'\n", descriptorSet.c_str());
	}

	return ret;
}

vk::Sampler& Renderer::getDefaultSampler() {
	return this->defaultSampler;
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

void Renderer::setRecreateSwapchain(bool value, bool force) {
	this->recreateSwapchain = value;
	this->forceRecreate = force;
}
