#include "Renderer.hpp"

#include <iostream>

#include "Error.hpp"
#include "toString.hpp"

#include "../Driver.hpp"
#include "../baked/BakedModel.hpp"
#include "../imgui/imgui.h"
#include "../imgui/backends/imgui_impl_vulkan.h"
#include "../imgui/backends/imgui_impl_glfw.h"

#include "PipelineCreation.hpp"

#include "objects/impl/renderPasses/ForwardPass.hpp"
#include "objects/impl/renderPasses/ShadowPass.hpp"
#include "objects/impl/renderPasses/CubemapShadowPass.hpp"
#include "objects/impl/renderPasses/GUIPass.hpp"

#include "objects/impl/pipelineLayouts/ForwardPipelineLayout.hpp"
#include "objects/impl/pipelineLayouts/ShadowPipelineLayout.hpp"
#include "objects/impl/pipelineLayouts/CubemapShadowPipelineLayout.hpp"

#include "objects/impl/pipelines/ForwardPipeline.hpp"
#include "objects/impl/pipelines/ShadowPipeline.hpp"
#include "objects/impl/pipelines/CubemapShadowPipeline.hpp"

#include "objects/impl/textureBuffers/DepthTextureBuffer.hpp"
#include "objects/impl/textureBuffers/ShadowDepthTextureBuffer.hpp"
#include "objects/impl/textureBuffers/CubemapDepthTextureBuffer.hpp"
#include "objects/impl/textureBuffers/ColourTextureBuffer.hpp"

#include "objects/impl/framebuffers/ForwardFramebuffer.hpp"
#include "objects/impl/framebuffers/ShadowFramebuffer.hpp"
#include "objects/impl/framebuffers/CubemapFramebuffer.hpp"
#include "objects/impl/framebuffers/GUIFramebuffer.hpp"

#include "objects/impl/uniformBuffers/MVPUniformBuffer.hpp"
#include "objects/impl/uniformBuffers/DepthMVPUniformBuffer.hpp"
#include "objects/impl/uniformBuffers/CameraPlanesUniformBuffer.hpp"

#include "objects/impl/descriptorSets/BufferDescriptorSet.hpp"
#include "objects/impl/descriptorSets/ImageDescriptorSet.hpp"

#include "../vulkan/VulkanDevice.hpp"
#include "RendererUtils.hpp"

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

Renderer::Renderer(Driver* driver) : driver(driver) {
	this->context.window = initialiseVulkanWindow();
	this->context.allocator = initialiseVulkanAllocator(*this->context.window);

	this->camera = Camera(90.0f, 0.01f, 1000.0f, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f));

	VulkanWindow* window = this->context.window.get();
	VulkanAllocator* allocator = this->context.allocator.get();

	// Render passes
	this->renderPasses.emplace("forward", std::make_unique<ForwardPass>(window, &this->sampleCountSetting));
	this->renderPasses.emplace("shadow", std::make_unique<ShadowPass>(window, &this->sampleCountSetting));
	this->renderPasses.emplace("cubemapShadow", std::make_unique<CubemapShadowPass>(window, &this->sampleCountSetting));
	this->renderPasses.emplace("gui", std::make_unique<GUIPass>(window, &this->sampleCountSetting));

	// Descriptor Set Layouts
	std::vector<DescriptorSetting> uniformBufferV = { { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT } };
	std::vector<DescriptorSetting> uniformBufferF = { { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT } };
	std::vector<DescriptorSetting> uniformBufferVF = { { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT }};
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
	this->descriptorSetLayouts.emplace("imageF", createDescriptorLayout(*window, imageF));
	this->descriptorSetLayouts.emplace("materials", createDescriptorLayout(*window, materialSettings));

	// Pipeline Layouts
	this->pipelineLayouts.emplace("forward", std::make_unique<ForwardPipelineLayout>(window, &this->descriptorSetLayouts, &this->shadowsEnabled));
	this->pipelineLayouts.emplace("shadow", std::make_unique<ShadowPipelineLayout>(window, &this->descriptorSetLayouts));
	this->pipelineLayouts.emplace("cubemapShadow", std::make_unique<CubemapShadowPipelineLayout>(window, &this->descriptorSetLayouts));

	// Pipelines
	this->pipelines.emplace("forward", std::make_unique<ForwardPipeline>(window, &this->pipelineLayouts.at("forward"), &this->renderPasses.at("forward"), &this->sampleCountSetting, &this->shadowsEnabled));
	this->pipelines.emplace("shadow", std::make_unique<ShadowPipeline>(window, &this->pipelineLayouts.at("shadow"), &this->renderPasses.at("shadow"), &this->sampleCountSetting, &this->currentShadowResolution));
	this->pipelines.emplace("cubemapShadow", std::make_unique<CubemapShadowPipeline>(window, &this->pipelineLayouts.at("cubemapShadow"), &this->renderPasses.at("cubemapShadow"), &this->sampleCountSetting, &this->currentShadowResolution));

	// Texture Buffers
	this->textureBuffers.emplace("depth", std::make_unique<DepthTextureBuffer>(&this->context, &this->sampleCountSetting));
	this->textureBuffers.emplace("shadowDepth", std::make_unique<ShadowDepthTextureBuffer>(&this->context, &this->sampleCountSetting, &this->currentShadowResolution));
	this->textureBuffers.emplace("cubemapDepth", std::make_unique<CubemapDepthTextureBuffer>(&this->context, &this->sampleCountSetting, &this->currentShadowResolution));
	this->textureBuffers.emplace("debugLinearDepth", std::make_unique<ColourTextureBuffer>(&this->context, &this->sampleCountSetting, &this->currentShadowResolution));

	// Framebuffers
	this->framebuffers.emplace("forward", std::make_unique<ForwardFramebuffer>(window, &this->textureBuffers, &this->renderPasses.at("forward"), &this->sampleCountSetting));
	this->framebuffers.emplace("shadow", std::make_unique<ShadowFramebuffer>(window, &this->textureBuffers, &this->renderPasses.at("shadow"), &this->currentShadowResolution));
	this->framebuffers.emplace("cubemapShadow", std::make_unique<CubemapFramebuffer>(window, &this->textureBuffers, &this->renderPasses.at("cubemapShadow"), &this->currentShadowResolution));
	this->framebuffers.emplace("gui", std::make_unique<GUIFramebuffer>(window, &this->renderPasses.at("gui")));

	// Uniform Buffers
	VkPipelineStageFlags VFstageFlags = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	VkPipelineStageFlags VstageFlags = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT;
	VkPipelineStageFlags FstageFlags = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

	this->uniformBuffers.emplace("mvp", std::make_unique<MVPUniformBuffer>(allocator, VFstageFlags, &this->uniforms.mvpUniform));
	this->uniformBuffers.emplace("depthMVP", std::make_unique<DepthMVPUniformBuffer>(allocator, VstageFlags, &this->uniforms.depthMVPUniform));
	this->uniformBuffers.emplace("cameraPlanes", std::make_unique<CameraPlanesUniformBuffer>(allocator, FstageFlags, &this->uniforms.cameraPlanesUniform));

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
		VK_SAMPLER_ADDRESS_MODE_REPEAT};
	this->defaultSampler = createTextureSampler(*window, defaultSamplerInfo);
	SamplerInfo shadowMapSamplerInfo = {
		VK_FILTER_LINEAR,
		VK_FILTER_LINEAR,
		VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		1, VK_COMPARE_OP_LESS_OR_EQUAL };
	this->shadowMapSampler = createTextureSampler(*window, shadowMapSamplerInfo);

	// Descriptor Sets
	std::vector<DescriptorBufferSetting> mvpDescriptorSettings = {{ this->uniformBuffers.at("mvp").get(), VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER }};
	std::vector<DescriptorBufferSetting> depthDescriptorSettings = {{ this->uniformBuffers.at("depthMVP").get(), VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER }};
	std::vector<DescriptorBufferSetting> cameraPlanesDescriptorSettings = {{ this->uniformBuffers.at("cameraPlanes").get(), VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER }};
	std::vector<DescriptorImageSetting> shadowMapDescriptorSettings = { 
		{ this->textureBuffers.at("shadowDepth").get(), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, this->shadowMapSampler.handle }};
	std::vector<DescriptorImageSetting> debugLinearDepthDescriptorSettings = { 
		{ this->textureBuffers.at("debugLinearDepth").get(), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, this->defaultSampler.handle }};
	std::vector<DescriptorImageSetting> shadowCubemapDescriptorSettings = {
		{ this->textureBuffers.at("cubemapDepth").get(), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, this->shadowMapSampler.handle }};

	this->descriptorSets.emplace("mvp", std::make_unique<BufferDescriptorSet>(window, &this->descriptorSetLayouts.at("uboVF").handle, mvpDescriptorSettings));
	this->descriptorSets.emplace("depthMVP", std::make_unique<BufferDescriptorSet>(window, &this->descriptorSetLayouts.at("uboV").handle, depthDescriptorSettings));
	this->descriptorSets.emplace("cameraPlanes", std::make_unique<BufferDescriptorSet>(window, &this->descriptorSetLayouts.at("uboF").handle, cameraPlanesDescriptorSettings));
	this->descriptorSets.emplace("shadowMap", std::make_unique<ImageDescriptorSet>(window, &this->descriptorSetLayouts.at("imageF").handle, shadowMapDescriptorSettings));
	this->descriptorSets.emplace("debugLinearDepth", std::make_unique<ImageDescriptorSet>(window, &this->descriptorSetLayouts.at("imageF").handle, debugLinearDepthDescriptorSettings));
	this->descriptorSets.emplace("shadowCubemap", std::make_unique<ImageDescriptorSet>(window, &this->descriptorSetLayouts.at("imageF").handle, shadowCubemapDescriptorSettings));

	// TEMP
	glm::vec3 lightPos(-0.2972f, 7.3100f, -11.9532f);
	
	glm::mat4 cubePerspective = glm::perspective(glm::radians(90.0f), 1.0f, 0.01f, 256.0f);
	cubePerspective[1][1] *= -1.0f;

	glm::vec3 directions[6] = {
		glm::vec3(1.0f, 0.0f, 0.0f),
		glm::vec3(-1.0f, 0.0f, 0.0f),
		glm::vec3(0.0f, 1.0f, 0.0f),
		glm::vec3(0.0f, -1.0f, 0.0f),
		glm::vec3(0.0f, 0.0f, 1.0f),
		glm::vec3(0.0f, 0.0f, -1.0f),
	};

	glm::vec3 upVectors[6] = {
		glm::vec3(0.0f, 1.0f, 0.0f),
		glm::vec3(0.0f, 1.0f, 0.0f),
		glm::vec3(0.0f, 0.0f, 1.0f),
		glm::vec3(0.0f, 0.0f, -1.0f),
		glm::vec3(0.0f, 1.0f, 0.0f),
		glm::vec3(0.0f, 1.0f, 0.0f),
	};
	
	for (std::size_t i = 0; i < 6; i++) {
		glm::mat4 cubeView = glm::lookAt(lightPos, lightPos + directions[i], upVectors[i]);
		this->cubeProjections.emplace_back(cubePerspective * cubeView);
	}
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

	float width = this->context.window->swapchainExtent.width;
	float height = this->context.window->swapchainExtent.height;
	const float aspectRatio = width / height;

	this->uniforms.mvpUniform.projection = glm::perspective(
		glm::radians(this->camera.getFov()), 
		aspectRatio, 
		this->camera.getNearPlane(), 
		this->camera.getFarPlane());
	this->uniforms.mvpUniform.projection[1][1] *= -1.0f;
	this->uniforms.mvpUniform.view = glm::lookAt(
		this->camera.getPosition(), 
		this->camera.getPosition() + this->camera.getFrontDir(), 
		glm::vec3(0.0f, 1.0f, 0.0f));
	this->uniforms.mvpUniform.camPos = glm::vec4(this->camera.getPosition(), 1.0f);

	//glm::mat4 depthProjection = glm::ortho(-10.0f, 10.0f, 10.0f, -10.0f, 0.1f, 1000.0f);
	glm::mat4 depthProjection = glm::perspective(glm::radians(90.0f), 1.0f, 0.01f, 256.0f);
	depthProjection[1][1] *= -1.0f;
	glm::mat4 depthView = glm::lookAt(
		glm::vec3(-0.2972f, 7.3100f, -11.9532f),
		glm::vec3(0.0f, 0.0f, -48.0f),
		glm::vec3(0.0f, 1.0f, 0.0f));

	this->uniforms.depthMVPUniform.depthMVP = depthProjection * depthView;
	this->uniforms.cameraPlanesUniform._far = this->camera.getFarPlane();
	this->uniforms.cameraPlanesUniform._near = this->camera.getNearPlane();
}

void Renderer::render() {
	// Begin command buffer
	VkCommandBuffer cmdBuff = this->cmdBuffers[this->frameIndex];
	beginCommandBuffer(cmdBuff, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

	std::vector<MeshData>& meshData = this->driver->getMeshData();

	// Update uniform buffers
	this->uniformBuffers.at("mvp")->update(cmdBuff);

	// Shadow pass
	if (this->shadowsEnabled) {

		// Render to each face of the cube map
		for (std::size_t face = 0; face < 6; face++) {
			this->uniforms.depthMVPUniform.depthMVP = this->cubeProjections[face];
			this->uniformBuffers.at("depthMVP")->update(cmdBuff);

			RendererUtils::beginRenderPass(cmdBuff, this->renderPasses.at("cubemapShadow").get(), this->framebuffers.at("cubemapShadow").get(), face);

			vkCmdBindPipeline(cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, this->pipelines.at("cubemapShadow")->getHandle());
			vkCmdBindDescriptorSets(
				cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS,
				this->pipelineLayouts.at("cubemapShadow")->getHandle(), 0, 1,
				&this->descriptorSets.at("depthMVP")->getHandle(), 0, nullptr);
			vkCmdSetDepthBias(cmdBuff, this->depthBiasConstant, 0.0f, this->depthBiasSlopeFactor);

			for (std::size_t i = 0; i < meshData.size(); i++)
				this->drawMeshGeometry(cmdBuff, meshData[i]);

			RendererUtils::endRenderPass(cmdBuff);
		}

		//this->uniformBuffers.at("depthMVP")->update(cmdBuff);
		//this->uniformBuffers.at("cameraPlanes")->update(cmdBuff);

		//RendererUtils::beginRenderPass(cmdBuff, this->renderPasses.at("shadow").get(), this->framebuffers.at("shadow").get(), this->imageIndex);

		//vkCmdBindPipeline(cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, this->pipelines.at("shadow")->getHandle());
		//vkCmdBindDescriptorSets(
		//	cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS,
		//	this->pipelineLayouts.at("shadow")->getHandle(), 0, 1,
		//	&this->descriptorSets.at("depthMVP")->getHandle(), 0, nullptr);
		//vkCmdBindDescriptorSets(
		//	cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS,
		//	this->pipelineLayouts.at("shadow")->getHandle(), 1, 1,
		//	&this->descriptorSets.at("cameraPlanes")->getHandle(), 0, nullptr);
		//vkCmdSetDepthBias(cmdBuff, this->depthBiasConstant, 0.0f, this->depthBiasSlopeFactor);

		//for (std::size_t i = 0; i < meshData.size(); i++)
		//	this->drawMeshGeometry(cmdBuff, meshData[i]);

		//RendererUtils::endRenderPass(cmdBuff);
	}

	// Forward pass
	RendererUtils::beginRenderPass(cmdBuff, this->renderPasses.at("forward").get(), this->framebuffers.at("forward").get(), this->imageIndex);

	vkCmdBindPipeline(cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, this->pipelines.at("forward")->getHandle());
	vkCmdBindDescriptorSets(
		cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, 
		this->pipelineLayouts.at("forward")->getHandle(), 0, 1, 
		&this->descriptorSets.at("mvp")->getHandle(), 0, nullptr);

	if (this->shadowsEnabled) {
		vkCmdBindDescriptorSets(cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS,
			this->pipelineLayouts.at("forward")->getHandle(), 2, 1,
			&this->descriptorSets.at("depthMVP")->getHandle(), 0, nullptr);
		vkCmdBindDescriptorSets(cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS,
			this->pipelineLayouts.at("forward")->getHandle(), 3, 1,
			&this->descriptorSets.at("shadowMap")->getHandle(), 0, nullptr);
	}

	vkCmdSetCullMode(cmdBuff, VK_CULL_MODE_BACK_BIT);

	auto perMeshCallback = [this](VkCommandBuffer cmdBuff, MeshData& meshData) {
		vkCmdBindDescriptorSets(
			cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS,
			this->pipelineLayouts.at("forward")->getHandle(), 1, 1,
			&this->driver->getMaterialDescriptors().at(meshData.materialId), 0, nullptr);
	};

	// Draw non-alpha masked meshes
	for (std::size_t i = 0; i < meshData.size(); i++) {
		if (meshData[i].hasAlphaMask) continue;

		this->drawMesh(cmdBuff, meshData[i], perMeshCallback);
	}

	vkCmdSetCullMode(cmdBuff, VK_CULL_MODE_NONE);

	// Draw alpha masked meshes
	for (std::uint32_t i = 0; i < meshData.size(); i++) {
		if (!meshData[i].hasAlphaMask) continue;

		this->drawMesh(cmdBuff, meshData[i], perMeshCallback);
	}

	RendererUtils::endRenderPass(cmdBuff);

	RendererUtils::beginRenderPass(cmdBuff, this->renderPasses.at("gui").get(), this->framebuffers.at("gui").get(), this->imageIndex);

	if (ImGui::GetDrawData() != nullptr)
		ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmdBuff);

	RendererUtils::endRenderPass(cmdBuff);

	endCommandBuffer(*this->context.window, cmdBuff);
}

void Renderer::drawMesh(VkCommandBuffer cmdBuff, MeshData& meshData, const std::function<void(VkCommandBuffer, MeshData&)>& perMeshCallback) {
	if (perMeshCallback)
		perMeshCallback(cmdBuff, meshData);
	
	VkBuffer vBuffers[3] = { 
		meshData.posBuffer.buffer,
		meshData.texCoordBuffer.buffer,
		meshData.tbnFrameBuffer.buffer
	};
	VkBuffer iBuffer = meshData.indicesBuffer.buffer;
	VkDeviceSize vOffsets[3]{};
	VkDeviceSize iOffset{};

	vkCmdBindVertexBuffers(cmdBuff, 0, 3, vBuffers, vOffsets);
	vkCmdBindIndexBuffer(cmdBuff, iBuffer, iOffset, VK_INDEX_TYPE_UINT32);

	vkCmdDrawIndexed(cmdBuff, meshData.indicesCount, 1, 0, 0, 0);
}

void Renderer::drawMeshGeometry(VkCommandBuffer cmdBuff, MeshData& meshData, const std::function<void(VkCommandBuffer, MeshData&)>& perMeshCallback) {
	if (perMeshCallback)
		perMeshCallback(cmdBuff, meshData);

	VkBuffer vBuffer = meshData.posBuffer.buffer;
	VkBuffer iBuffer = meshData.indicesBuffer.buffer;
	VkDeviceSize vOffset{};
	VkDeviceSize iOffset{};

	vkCmdBindVertexBuffers(cmdBuff, 0, 1, &vBuffer, &vOffset);
	vkCmdBindIndexBuffer(cmdBuff, iBuffer, iOffset, VK_INDEX_TYPE_UINT32);

	vkCmdDrawIndexed(cmdBuff, meshData.indicesCount, 1, 0, 0, 0);
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
	} catch (const std::out_of_range& ex) {
		std::printf("Could not find: %s in 'textureBuffers'\n", textureBuffer.c_str());
	}

	return ret;
}

DescriptorSet* Renderer::getDescriptorSet(const std::string& descriptorSet) {
	DescriptorSet* ret = nullptr;

	try {
		ret = this->descriptorSets.at(descriptorSet).get();
	} catch (const std::out_of_range& ex) {
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
