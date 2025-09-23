#pragma once

#include <map>
#include <string>
#include <cstdio>

#include "Uniforms.hpp"
#include "lights/Light.hpp"
#include "../baked/BakedModel.hpp"
#include "../camera/Camera.hpp"
#include "../vulkan/VulkanContext.hpp"
#include "objects/base/interfaces/IShaderStorageBuffer.hpp"
#include "objects/base/interfaces/IUniformBuffer.hpp"
#include "objects/base/RenderPass.hpp"
#include "objects/base/PipelineLayout.hpp"
#include "objects/base/Pipeline.hpp"
#include "objects/base/Framebuffer.hpp"
#include "objects/base/TextureBuffer.hpp"
#include "objects/base/UniformBuffer.hpp"
#include "objects/base/ShaderStorageBuffer.hpp"
#include "objects/base/DescriptorSet.hpp"

struct Uniforms {
	glsl::MVPUniform mvpUniform;
	glsl::CameraPlanesUniform cameraPlanesUniform;
};

struct SSBOs {
	std::vector<glsl::Light> lights;
	std::vector<glm::mat4> lightMatrices;
};

struct LightMatrices {
	glm::mat4 projection;
	glm::mat4 view;
};

class Driver;

using _RenderPass = std::unique_ptr<RenderPass>;
using _PipelineLayout = std::unique_ptr<PipelineLayout>;
using _Pipeline = std::unique_ptr<Pipeline>;
using _Framebuffer = std::unique_ptr<Framebuffer>;
using _TextureBuffer = std::unique_ptr<TextureBuffer>;
using _UniformBuffer = std::unique_ptr<IUniformBuffer>;
using _ShaderStorageBuffer = std::unique_ptr<IShaderStorageBuffer>;
using _DescriptorSet = std::unique_ptr<DescriptorSet>;

class Renderer {
public:
	Renderer() = default;
	Renderer(Driver* driver);
	~Renderer();

	Renderer(const Renderer&) = delete;
	Renderer& operator=(const Renderer&) = delete;
	Renderer(Renderer&&) = delete;
	Renderer& operator=(Renderer&&) = delete;

	void setLights(std::vector<Light>* lights);

	// Swapchain
	bool checkSwapchain();
	bool acquireSwapchainImage();

	// Updates
	void update(float timeDelta);

	void render();

	void submitRender();
	void finishRendering();

	// Recreation
	void recreateFormatDependents();
	void recreateSizeDependents();

	VulkanContext& getContext();
	Camera& getCamera();

	VkRenderPass getRenderPassHandle(const std::string& renderPass);
	std::map<std::string, vk::DescriptorSetLayout>& getDescriptorSetLayouts();
	TextureBuffer* getTextureBuffer(const std::string& textureBuffer);
	DescriptorSet* getDescriptorSet(const std::string& descriptorSet);
	vk::Sampler& getDefaultSampler();

	Uniforms& getUniforms();
	SSBOs& getSSBOs();

	int& getRenderingType();
	bool& getShadowsEnabled();
	float& getDepthBiasConstant();
	float& getDepthBiasSlopeFactor();
	bool& getDebugView();
	int& getDebugState();

	void setRecreateSwapchain(bool value, bool force = false);

	int numLights = 0;
	std::uint32_t numPointLights = 0;
	std::uint32_t numDirectionalLights = 0;
	std::uint32_t numSpotLights = 0;

	float sunOrthoBounds = 20.0f;
	float sunShadowNear = 0.1f;
	float sunShadowFar = 256.0f;
	float sunDistance = 50.0f;

	bool renderCameraFrustumBounds = false;
	float zMult = 10.0f;
private:
	void renderForward();
	void renderDeferred();
	void renderDebugViews();
	void renderShadowMaps();
	LightMatrices getLightMatricesForCameraFrustum(glsl::Light& lightStruct);
	LightMatrices getSunViewMatrices(glsl::Light& lightStruct);

	Driver* driver;
	VulkanContext context;

	Camera camera;

	// Vulkan object maps
	std::map<std::string, _RenderPass> renderPasses;
	std::map<std::string, vk::DescriptorSetLayout> descriptorSetLayouts;
	std::map<std::string, _PipelineLayout> pipelineLayouts;
	std::map<std::string, _Pipeline> pipelines;
	std::map<std::string, _Framebuffer> framebuffers;
	std::map<std::string, _TextureBuffer> textureBuffers;
	std::map<std::string, _UniformBuffer> uniformBuffers;
	std::map<std::string, _ShaderStorageBuffer> shaderStorageBuffers;
	std::map<std::string, _DescriptorSet> descriptorSets;

	// Synchronisation variables
	std::uint32_t frameIndex = 0;
	std::uint32_t imageIndex = 0;
	std::vector<VkCommandBuffer> cmdBuffers;
	std::vector<vk::Fence> frameDoneFences;
	std::vector<vk::Semaphore> imageAvailableSemaphores;
	std::vector<vk::Semaphore> renderFinishedSemaphores;

	// Samplers
	vk::Sampler defaultSampler;
	vk::Sampler shadowMapSampler;

	// Lights pointer
	std::vector<Light>* lights = nullptr;

	// Shader objects
	Uniforms uniforms;
	SSBOs ssbos;

	// Renderer settings
	int renderingType = 0; // 0 = Forward, 1 = Deferred
	VkSampleCountFlagBits sampleCountSetting = VK_SAMPLE_COUNT_1_BIT;
	bool shadowsEnabled = true;
	VkExtent2D shadowRes = VkExtent2D{ 2048, 2048 };
	std::vector<VkExtent2D> shadowResolutions = {
		VkExtent2D{ 1024, 1024 },
		VkExtent2D{ 2048, 2048 },
		VkExtent2D{ 4096, 4096 },
		VkExtent2D{ 8192, 8192 }
	};
	float depthBiasConstant = 7.0f;
	float depthBiasSlopeFactor = 8.0f;
	bool debugView = false;
	int debugState = 0;

	// Internal
	bool recreateSwapchain = false;
	bool forceRecreate = false;

	LightMatrices sunMatrices;
	std::size_t sunLightIndex = -1;

	LineMeshData lineMeshData;
	bool lineMeshDataInit = false;

	// Shutdown logic
	bool handledImGUIShutdown = false;
};