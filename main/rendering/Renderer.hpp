#pragma once

#include <map>
#include <string>
#include <cstdio>

#include "Uniforms.hpp"
#include "lights/Light.hpp"
#include "../baked/BakedModel.hpp"
#include "../camera/Camera.hpp"
#include "../vulkan/VulkanContext.hpp"
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
	glsl::DepthMVPUniform depthMVPUniform;
	glsl::CameraPlanesUniform cameraPlanesUniform;
};

struct SSBOs {
	std::vector<glsl::Light> lights;
	std::vector<glm::mat4> lightMatrices;
};

class Driver;

using _RenderPass = std::unique_ptr<RenderPass>;
using _PipelineLayout = std::unique_ptr<PipelineLayout>;
using _Pipeline = std::unique_ptr<Pipeline>;
using _Framebuffer = std::unique_ptr<Framebuffer>;
using _TextureBuffer = std::unique_ptr<TextureBuffer>;
using _UniformBuffer = std::unique_ptr<UniformBuffer>;
using _ShaderStorageBuffer = std::unique_ptr<ShaderStorageBuffer>;
using _DescriptorSet = std::unique_ptr<DescriptorSet>;

class Renderer {
public:
	Renderer() = default;
	Renderer(Driver* driver);

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
	void drawMesh(MeshData& meshData, const std::function<void(MeshData&)>& perMeshCallback = nullptr);
	void drawMeshGeometry(MeshData& meshData, const std::function<void(MeshData&)>& perMeshCallback = nullptr);

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
	bool& getShadowsEnabled();
	float& getDepthBiasConstant();
	float& getDepthBiasSlopeFactor();

	void setRecreateSwapchain(bool value, bool force = false);

	float shadowBias = 0.0001f;

	int numLights = 0;
	std::uint32_t numPointLights = 0;
	std::uint32_t numDirectionalLights = 0;
	std::uint32_t numSpotLights = 0;
private:
	void renderShadowMaps(std::vector<MeshData>& meshData);

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

	// Internal
	VkCommandBuffer cmdBuff = VK_NULL_HANDLE;
	bool recreateSwapchain = false;
	bool forceRecreate = false;

	VkExtent2D dummyExtent{ 1, 1 };
};