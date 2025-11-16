#pragma once

#include <map>
#include <string>
#include <cstdio>

#include "Uniforms.hpp"
#include "Cache.hpp"
#include "lights/Light.hpp"
#include "../baked/BakedModel.hpp"
#include "../camera/Camera.hpp"
#include "../models/OBJLoader.hpp"
#include "../models/MeshData.hpp"
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
#include "objects/base/PostProcessingEffect.hpp"

#include "preProcess/SSAOPreProcess.hpp"

#include <stb_image.h>

struct Uniforms {
	glsl::MVPUniform mvpUniform;
	glsl::CameraPlanesUniform cameraPlanesUniform;
	glsl::ProjectiveUniform projectiveUniform;
	glsl::InverseMatricesUniform inverseMatricesUniform;
	glsl::SSAOUniform ssaoUniform;
};

struct SSBOs {
	std::vector<glsl::Light> lights;
	std::vector<glm::mat4> lightMatrices;
};

struct LightMatrices {
	Cache<glm::mat4> projection;
	Cache<glm::mat4> view;
};

enum TapSize {
	e5X5,
	e9X9,
	e17X17,
	e25X25,
	e41X41
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
using _PostProcessingEffect = std::unique_ptr<PostProcessingEffect>;

using _SSAOPreProcess = std::unique_ptr<SSAOPreProcess>;

using VkDescriptorSetPair = std::pair<VkDescriptorSet, VkDescriptorSet>;
using WriteToFramebufferPair = std::pair<WriteToTargetFramebuffer*, WriteToTargetFramebuffer*>;

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
	void recreateSwapViewDependents();
	void setRecreateSwapchain(bool value, bool force = false);

	Driver* getDriver();
	VulkanContext& getContext();
	Camera* getCamera();

	RenderPass* getRenderPass(const std::string& renderPass);
	VkDescriptorSetLayout getDescriptorSetLayout(const std::string& descriptorSetLayout);
	PipelineLayout* getPipelineLayout(const std::string& pipelineLayout);
	Pipeline* getPipeline(const std::string& pipeline);
	Framebuffer* getFramebuffer(const std::string& framebuffer);
	TextureBuffer* getTextureBuffer(const std::string& textureBuffer);
	IUniformBuffer* getUniformBuffer(const std::string& uniformBuffer);
	IShaderStorageBuffer* getShaderStorageBuffer(const std::string& shaderStorageBuffer);
	DescriptorSet* getDescriptorSet(const std::string& descriptorSet);

	SSAOPreProcess* getSSAOPreProcess();

	std::vector<std::pair<std::string, _PostProcessingEffect>>& getPostProcessingEffects();

	vk::Sampler& getDefaultSampler();

	std::uint32_t getFrameIndex();
	std::uint32_t getImageIndex();

	Uniforms& getUniforms();
	SSBOs& getSSBOs();

	int& getRenderingType();
	bool& getShadowsEnabled();
	float& getDepthBiasConstant();
	float& getDepthBiasSlopeFactor();
	bool& getDebugView();
	int& getDebugState();

	// Post processing effects
	bool& getMosaicEnabled();

	std::pair<vk::Image, vk::ImageView>& getDummyTexture();
	LightMatrices& getSunMatrices();
	std::uint32_t getSunLightIndex();

	int numLights = 2;
	std::uint32_t numPointLights = 0;
	std::uint32_t numDirectionalLights = 0;
	std::uint32_t numSpotLights = 0;

	// Should only have 1 sun shadow map and is usually most important
	// so we use high resolution here
	int sunShadowMapResIdx = 3;
	int pointShadowMapResIdx = 1;
	int spotShadowMapResIdx = 1;

	bool showSunView = false;

	float emissiveStrength = 75.0f;

	int bloomIterations = 1;
	int bloomTapSize = TapSize::e41X41;
	float brightnessThreshold = 0.75f;
	
	float shadowBias = 0.00025f;
	float bleedReduction = 0.7f;
	int vsmShadowsEnabled = 1;
	int vsmTapSize = TapSize::e5X5;

	float sunOrthoBounds = 37.0f;
	float sunShadowNear = 0.1f;
	float sunShadowFar = 256.0f;
	float sunDistance = 100.0f;

	bool renderCameraFrustumBounds = false;
	
	bool ssaoEnabled = false;
	float ssaoExp = 2.0f;

	float sunUpperStep = 0.12f;
	float sunLowerStep = 0.01f;
	// Different to the 'intensity' of the light of the sun, 
	// this is to do with rendering the actual sun in the sky
	float sunIntensity = 1.0f;

	// Not the best implementation by just sticking this in
	// here but its the simplest
	std::vector<VkDescriptorSet> alphaMaskDescriptors;
private:
	void loadObjShapes();
	void createDummyTexture();

	void renderForward();
	void renderDeferred();
	void renderDebugViews();
	void renderShadowMaps();
	void renderVSMShadowMaps();
	void blurVSMShadowMap(
		Pipeline* blurPipeline,
		Framebuffer* writeToBlurFB, 
		Framebuffer* writeToMapFB, 
		DescriptorSet* shadowMapToRead1, 
		DescriptorSet* shadowMapToRead2, 
		std::uint32_t imageIndex);

	void getSkyboxDimensions(std::array<const char*, 6>& filenames);
	void fillSkyboxTexture();

	void setObjectDebugNames();

	Driver* driver;
	VulkanContext context;

	std::unique_ptr<Camera> camera;

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

	// Pre processing effects
	_SSAOPreProcess ssaoEffect;

	// Post processing effects
	std::vector<std::pair<std::string, _PostProcessingEffect>> postProcessingEffects;

	// Synchronisation variables
	std::uint32_t frameIndex = 0;
	std::uint32_t imageIndex = 0;
	std::vector<VkCommandBuffer> cmdBuffers;
	std::vector<vk::Fence> frameDoneFences;
	std::vector<vk::Semaphore> imageAvailableSemaphores;
	std::vector<vk::Semaphore> renderFinishedSemaphores;

	// Samplers
	vk::Sampler linearRepeatSampler;
	vk::Sampler linearMirroredRepeatSampler;
	vk::Sampler linearClampToEdgeSampler;
	vk::Sampler linearClampToBorderSampler;
	vk::Sampler nearestRepeatSampler;
	vk::Sampler nearestClampToEdgeSampler;
	vk::Sampler depthSampler;
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
	VkExtent2D sunShadowMapRes = VkExtent2D{ 8192, 8192 };
	VkExtent2D pointShadowMapRes = VkExtent2D{ 1024, 1024 };
	VkExtent2D spotShadowMapRes = VkExtent2D{ 2048, 2048 };
	std::array<VkExtent2D, 4> shadowResolutions = {
		VkExtent2D{ 1024, 1024 },
		VkExtent2D{ 2048, 2048 },
		VkExtent2D{ 4096, 4096 },
		VkExtent2D{ 8192, 8192 }
	};

	float depthBiasConstant = 7.0f;
	float depthBiasSlopeFactor = 3.0f;
	bool debugView = false;
	int debugState = 0;

	// Post processing effect states
	bool mosaicEnabled = false;

	// Internal
	MeshData sphereModel;
	std::pair<vk::Image, vk::ImageView> dummyTexture;

	stbi_uc* skyboxImageData[6];
	VkExtent2D skyboxDimensions;

	bool recreateSwapchain = false;
	bool forceRecreate = false;

	LightMatrices sunMatrices;
	std::size_t sunLightIndex = -1;

	LineMeshData lineMeshData;
	bool lineMeshDataInit = false;

	// Shutdown logic
	bool handledImGUIShutdown = false;
};