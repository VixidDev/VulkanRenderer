#include <tuple>
#include <chrono>
#include <limits>
#include <vector>
#include <stdexcept>
#include <iostream>

#include <cstdio>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include <algorithm>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <volk/volk.h>

#if !defined(GLM_FORCE_RADIANS)
#	define GLM_FORCE_RADIANS
#endif
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "../utils/to_string.hpp"
#include "../utils/vulkan_window.hpp"

#include "../utils/angle.hpp"
using namespace labutils::literals;

#include "../utils/error.hpp"
#include "../utils/vkutil.hpp"
#include "../utils/vkimage.hpp"
#include "../utils/vkobject.hpp"
#include "../utils/vkbuffer.hpp"
#include "../utils/allocator.hpp" 
namespace lut = labutils;

#include "baked_model.hpp"

// Anonymous namespace
namespace
{
	// Apologies for the very long file, I would have refactored it
	// and split it into separate files if I had the time, so all
	// I can recommend is to collapse all functions and only expand
	// functions when want to read one, or shift-click if you are
	// using an IDE that supports that.

	std::vector<lut::Image> images;

	namespace cfg {
		constexpr const char* kModelPath = "assets/main/suntemple.comp5892mesh";

		constexpr const char* kVertShaderPath = "assets/main/shaders/default.vert.spv";
		constexpr const char* kFragShaderPath = "assets/main/shaders/default.frag.spv"; 
		constexpr const char* kDebugVertShaderPath = "assets/main/shaders/debug.vert.spv";
		constexpr const char* kDebugFragShaderPath = "assets/main/shaders/debug.frag.spv";
		constexpr const char* kPPVertShaderPath = "assets/main/shaders/postProcess.vert.spv";
		constexpr const char* kPPFragShaderPath = "assets/main/shaders/postProcess.frag.spv";
		constexpr const char* kOverVisWriteVertShaderPath = "assets/main/shaders/overVisWrite.vert.spv";
		constexpr const char* kOverVisWriteFragShaderPath = "assets/main/shaders/overVisWrite.frag.spv";
		constexpr const char* kOverVisReadVertShaderPath = "assets/main/shaders/overVisRead.vert.spv";
		constexpr const char* kOverVisReadFragShaderPath = "assets/main/shaders/overVisRead.frag.spv";
		constexpr const char* kWriteGBufVertShaderPath = "assets/main/shaders/gBufWrite.vert.spv";
		constexpr const char* kWriteGBufFragShaderPath = "assets/main/shaders/gBufWrite.frag.spv";
		constexpr const char* kDefShadingVertShaderPath = "assets/main/shaders/deferredShading.vert.spv";
		constexpr const char* kDefShadingFragShaderPath = "assets/main/shaders/deferredShading.frag.spv";
		constexpr const char* kShadowOffscreenVertShaderPath = "assets/main/shaders/shadowOffscreen.vert.spv";
		constexpr const char* kShadowOffscreenFragShaderPath = "assets/main/shaders/shadowOffscreen.frag.spv";

		constexpr float kCameraNear = 0.1f;
		constexpr float kCameraFar = 100.f;

		constexpr auto kCameraFov = 60.0_degf;

		constexpr float kCameraBaseSpeed = 1.7f;
		constexpr float kCameraFastMult = 7.0f;
		constexpr float kCameraSlowMult = 0.05f;

		constexpr float kCameraMouseSensitivity = 0.01f;

		constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT_S8_UINT;
	}

	using Clock_ = std::chrono::steady_clock;
	using Secondsf_ = std::chrono::duration<float, std::ratio<1>>;

	// GLFW callbacks
	void glfw_callback_key_press(GLFWwindow*, int, int, int, int);
	void glfw_callback_button(GLFWwindow*, int, int, int);
	void glfw_callback_motion(GLFWwindow*, double, double);

	// Local types/structures:
	enum class EInputState {
		forward,
		backward,
		strafeLeft,
		strafeRight,
		levitate,
		sink,
		fast,
		slow,
		mousing,
		max
	};

	struct UserState {
		bool inputMap[std::size_t(EInputState::max)] = {};

		float mouseX = 0.0f, mouseY = 0.0f;
		float previousX = 0.0f, previousY = 0.0f;

		int debugVisualisation = 1;
		bool mosaicEffect = false;
		bool deferredShading = false;
		bool shadows = false;

		bool wasMousing = false;

		glm::mat4 camera2world = glm::identity<glm::mat4>();
	};

	// Structures to more easily pass them to record commands function
	struct MeshData {
		lut::Buffer positionBuffer;
		lut::Buffer texCoordBuffer;
		lut::Buffer normalsBuffer;
		lut::Buffer tangentsBuffer;
		lut::Buffer indicesBuffer;
		std::size_t indicesCount;
		std::uint32_t materialId;
		bool hasAlphaMask;
	};

	struct RenderPasses {
		VkRenderPass regularRenderPass;
		VkRenderPass offscreenRenderPass;
		VkRenderPass postProcessRenderPass;
		VkRenderPass overVisualisationsRenderPass;
		VkRenderPass deferredShadingRenderPass;
		VkRenderPass shadowOffscreenRenderPass;
	};

	struct Framebuffers {
		VkFramebuffer offscreenFramebuffer;
		VkFramebuffer regularSwapchainFramebuffer;
		VkFramebuffer fullscreenSwapchainFramebuffer;
		VkFramebuffer overVisualisationFramebuffer;
		VkFramebuffer deferredShadingFramebuffer;
		VkFramebuffer shadowOffscreenFramebuffer;
	};

	struct Pipelines {
		VkPipeline regularPipeline;
		VkPipeline alphaPipeline;
		VkPipeline alphaOffscreenPipeline;
		VkPipeline debugPipeline;
		VkPipeline offscreenPipeline;
		VkPipeline postProcessPipeline;
		VkPipeline overVisWritePipeline;
		VkPipeline overVisReadPipeline;
		VkPipeline gBufWritePipline;
		VkPipeline deferredShadingPipeline;
		VkPipeline shadowOffscreenPipeline;
	};

	struct UBOs {
		VkBuffer sceneUBO;
		VkBuffer lightUBO;
		VkBuffer debugUBO;
		VkBuffer multipleLightsUBO;
		VkBuffer depthMVPUBO;
	};

	struct PipelineLayouts {
		VkPipelineLayout regularPipelineLayout;
		VkPipelineLayout postProcessPipelineLayout;
		VkPipelineLayout overVisWritePipelineLayout;
		VkPipelineLayout overVisReadPipelineLayout;
		VkPipelineLayout gBufWritePipelineLayout;
		VkPipelineLayout deferredShadingPipelineLayout;
		VkPipelineLayout shadowOffscreenPipelineLayout;
	};

	struct DescriptorSets {
		std::vector<VkDescriptorSet>& materialDescriptors;
		VkDescriptorSet sceneDescriptors;
		VkDescriptorSet lightDescriptor;
		VkDescriptorSet debugDescriptor;
		VkDescriptorSet postProcessDescriptor;
		VkDescriptorSet overVisualisationDescriptor;
		VkDescriptorSet deferredShadingDescriptor;
		VkDescriptorSet multipleLightsDescriptor;
		VkDescriptorSet depthMVPDescriptor;
		VkDescriptorSet shadowMapDescriptor;
	};

	// Uniform data
	namespace glsl
	{
		struct SceneUniform {
			glm::mat4 camera;
			glm::mat4 projection;
			glm::mat4 projCam;
			glm::vec4 camPos;
		};

		struct LightUniform {
			glm::vec4 lightPos;
			glm::vec4 lightColour;
		};

		struct DebugUniform {
			int debug;
		};

		struct MultipleLights {
			LightUniform light[21];
		};

		struct DepthMVP {
			glm::mat4 depthMVP;
		};

		static_assert(sizeof(SceneUniform) <= 65536, "SceneUniform must be less than 65536 bytes for vkCmdUpdateBuffer");
		static_assert(sizeof(SceneUniform) % 4 == 0, "SceneUniform size must be a multiple of 4 bytes");
		static_assert(sizeof(LightUniform) % 4 == 0, "LightUniform size must be a multiple of 4 bytes");
	}

	struct Uniforms {
		glsl::SceneUniform sceneUniforms;
		glsl::LightUniform lightUniforms;
		glsl::DebugUniform debugUniforms;
		glsl::MultipleLights multipleLightsUniform;
		glsl::DepthMVP depthMVPUniform;
	};

	// Method declarations

	// Render Passes
	lut::RenderPass create_render_pass(const lut::VulkanWindow&);
	lut::RenderPass create_offscreen_render_pass(const lut::VulkanWindow&);
	lut::RenderPass create_post_process_render_pass(const lut::VulkanWindow&);
	lut::RenderPass create_over_visualisations_render_pass(const lut::VulkanWindow&);
	lut::RenderPass create_deferred_shading_render_pass(const lut::VulkanWindow&);
	lut::RenderPass create_offscreen_shadow_render_pass(const lut::VulkanWindow&);

	// Descriptor Set Layouts
	lut::DescriptorSetLayout create_scene_descriptor_layout(const lut::VulkanWindow&);
	lut::DescriptorSetLayout create_material_descriptor_layout(const lut::VulkanWindow&);
	lut::DescriptorSetLayout create_fragment_ubo_descriptor_layout(const lut::VulkanWindow&);
	lut::DescriptorSetLayout create_vertex_ubo_descriptor_layout(const lut::VulkanWindow&);
	lut::DescriptorSetLayout create_post_process_descriptor_layout(const lut::VulkanWindow&);
	lut::DescriptorSetLayout create_over_visualisations_descriptor_layout(const lut::VulkanWindow&);
	lut::DescriptorSetLayout create_deferred_shading_descriptor_layout(const lut::VulkanWindow&);
	lut::DescriptorSetLayout create_fragment_image_layout(const lut::VulkanWindow&);

	// Pipeline Layouts
	lut::PipelineLayout create_pipeline_layout(const lut::VulkanWindow&, std::vector<VkDescriptorSetLayout>&);

	// Piplines
	lut::Pipeline create_pipeline(const lut::VulkanWindow&, VkRenderPass, VkPipelineLayout);
	lut::Pipeline create_debug_pipeline(const lut::VulkanWindow&, VkRenderPass, VkPipelineLayout);
	lut::Pipeline create_alpha_pipeline(const lut::VulkanWindow&, VkRenderPass, VkPipelineLayout);
	lut::Pipeline create_post_process_pipeline(const lut::VulkanWindow&, VkRenderPass, VkPipelineLayout);
	std::tuple<lut::Pipeline, lut::Pipeline> create_over_visualisations_pipeline(const lut::VulkanWindow&, VkRenderPass, VkPipelineLayout, VkPipelineLayout);
	std::tuple<lut::Pipeline, lut::Pipeline> create_deferred_shading_pipeline(const lut::VulkanWindow&, VkRenderPass, VkPipelineLayout, VkPipelineLayout);
	lut::Pipeline create_shadow_pipeline(const lut::VulkanWindow&, VkRenderPass, VkPipelineLayout);

	// Buffers
	std::tuple<lut::Image, lut::ImageView> create_depth_buffer(const lut::VulkanWindow&, const lut::Allocator&, VkImageAspectFlagBits);
	std::tuple<lut::Image, lut::ImageView> create_colour_buffer(const lut::VulkanWindow&, const lut::Allocator&);
	std::tuple<lut::Image, lut::ImageView> create_just_stencil_buffer(const lut::VulkanWindow&, const lut::Allocator&);
	std::tuple<lut::Image, lut::ImageView> create_normals_buffer(const lut::VulkanWindow&, const lut::Allocator&);
	std::tuple<lut::Image, lut::ImageView> create_albedo_buffer(const lut::VulkanWindow&, const lut::Allocator&);
	std::tuple<lut::Image, lut::ImageView> create_shadow_depth_buffer(const lut::VulkanWindow&, const lut::Allocator&);

	// Framebuffers
	lut::Framebuffer create_offscreen_framebuffer(const lut::VulkanWindow&, VkRenderPass, VkImageView, VkImageView);
	void create_regular_swapchain_framebuffers(const lut::VulkanWindow&, VkRenderPass, std::vector<lut::Framebuffer>&, VkImageView);
	void create_fullscreen_swapchain_framebuffers(const lut::VulkanWindow&, VkRenderPass, std::vector<lut::Framebuffer>&);
	void create_over_visualisation_framebuffers(const lut::VulkanWindow&, VkRenderPass, std::vector<lut::Framebuffer>&, VkImageView, VkImageView);
	void create_deferred_shading_framebuffers(const lut::VulkanWindow&, VkRenderPass, std::vector<lut::Framebuffer>&, VkImageView, VkImageView, VkImageView);
	void create_shadow_offscreen_framebuffers(const lut::VulkanWindow&, VkRenderPass, std::vector<lut::Framebuffer>&, VkImageView);

	lut::ImageView load_mesh_texture(const lut::VulkanWindow&, VkCommandPool, const lut::Allocator&, BakedTextureInfo);
	lut::ImageView get_dummy_texture(const lut::VulkanWindow&, VkCommandPool, const lut::Allocator&);

	void update_user_state(UserState&, float);
	void update_scene_uniforms(glsl::SceneUniform&, std::uint32_t, std::uint32_t, const UserState&);
	void update_debug_uniforms(glsl::DebugUniform&, const UserState&);
	void update_depth_mvp_uniforms(glsl::DepthMVP&, std::uint32_t, std::uint32_t, glm::vec4);

	void record_commands(
		VkCommandBuffer aCmdBuff,
		RenderPasses aRenderPasses,
		Framebuffers aFramebuffers,
		Pipelines aPipelines,
		const VkExtent2D& aExtent,
		std::vector<MeshData>& aMeshData,
		UBOs aUBOs,
		Uniforms aUniforms,
		PipelineLayouts aPipelineLayouts,
		DescriptorSets aDescriptorSets,
		const UserState& aState
	);

	void submit_commands(const lut::VulkanWindow&, VkCommandBuffer, VkFence, VkSemaphore, VkSemaphore);
}

int main() try
{
	// Create Vulkan window
	lut::VulkanWindow window = lut::make_vulkan_window();

	// Zero initialise user state
	UserState state{};
	glfwSetWindowUserPointer(window.window, &state);

	// Configure GLFW callbacks
	glfwSetKeyCallback(window.window, &glfw_callback_key_press);
	glfwSetMouseButtonCallback(window.window, &glfw_callback_button);
	glfwSetCursorPosCallback(window.window, &glfw_callback_motion);

	// Create VMA allocator
	lut::Allocator allocator = lut::create_allocator(window);

	// Create render passes
	lut::RenderPass renderPass = create_render_pass(window);
	lut::RenderPass offscreenRenderPass = create_offscreen_render_pass(window);
	lut::RenderPass postProcessRenderPass = create_post_process_render_pass(window);
	lut::RenderPass overVisualisationsRenderPass = create_over_visualisations_render_pass(window);
	lut::RenderPass deferredShadingRenderPass = create_deferred_shading_render_pass(window);
	lut::RenderPass shadowOffscreenRenderPass = create_offscreen_shadow_render_pass(window);

	RenderPasses renderPasses{};
	renderPasses.regularRenderPass = renderPass.handle;
	renderPasses.offscreenRenderPass = offscreenRenderPass.handle;
	renderPasses.postProcessRenderPass = postProcessRenderPass.handle;
	renderPasses.overVisualisationsRenderPass = overVisualisationsRenderPass.handle;
	renderPasses.deferredShadingRenderPass = deferredShadingRenderPass.handle;
	renderPasses.shadowOffscreenRenderPass = shadowOffscreenRenderPass.handle;

	// Create descriptor set layouts
	lut::DescriptorSetLayout sceneLayout = create_scene_descriptor_layout(window);
	lut::DescriptorSetLayout materialLayout = create_material_descriptor_layout(window);
	lut::DescriptorSetLayout uboLayout = create_fragment_ubo_descriptor_layout(window);
	lut::DescriptorSetLayout uboLayoutVert = create_vertex_ubo_descriptor_layout(window);
	lut::DescriptorSetLayout postProcessDescriptorLayout = create_post_process_descriptor_layout(window);
	lut::DescriptorSetLayout overVisualisationDescriptorLayout = create_over_visualisations_descriptor_layout(window);
	lut::DescriptorSetLayout deferredShadingDescriptorLayout = create_deferred_shading_descriptor_layout(window);
	lut::DescriptorSetLayout fragImageLayout = create_fragment_image_layout(window);

	std::vector<VkDescriptorSetLayout> sceneDescriptorSetLayouts;
	sceneDescriptorSetLayouts.emplace_back(sceneLayout.handle);
	sceneDescriptorSetLayouts.emplace_back(materialLayout.handle);
	sceneDescriptorSetLayouts.emplace_back(uboLayout.handle);
	sceneDescriptorSetLayouts.emplace_back(uboLayoutVert.handle);
	sceneDescriptorSetLayouts.emplace_back(fragImageLayout.handle);

	std::vector<VkDescriptorSetLayout> postProcessDescriptorSetLayouts;
	postProcessDescriptorSetLayouts.emplace_back(postProcessDescriptorLayout.handle);

	std::vector<VkDescriptorSetLayout> overVisWriteDescriptorSetLayouts;
	overVisWriteDescriptorSetLayouts.emplace_back(sceneLayout.handle);

	std::vector<VkDescriptorSetLayout> overVisReadDescriptorSetLayouts;
	overVisReadDescriptorSetLayouts.emplace_back(overVisualisationDescriptorLayout.handle);

	std::vector<VkDescriptorSetLayout> gBufWriteDescriptorSetLayouts;
	gBufWriteDescriptorSetLayouts.emplace_back(sceneLayout.handle);

	std::vector<VkDescriptorSetLayout> deferredShadingDescriptorSetLayouts;
	deferredShadingDescriptorSetLayouts.emplace_back(deferredShadingDescriptorLayout.handle);
	deferredShadingDescriptorSetLayouts.emplace_back(sceneLayout.handle);
	deferredShadingDescriptorSetLayouts.emplace_back(uboLayout.handle);

	std::vector<VkDescriptorSetLayout> shadowOffscreenDescriptorSetLayouts;
	shadowOffscreenDescriptorSetLayouts.emplace_back(uboLayoutVert.handle);

	// Create pipeline layouts
	lut::PipelineLayout pipeLayout = create_pipeline_layout(window, sceneDescriptorSetLayouts);
	lut::PipelineLayout debugPipeLayout = create_pipeline_layout(window, sceneDescriptorSetLayouts);
	lut::PipelineLayout postProcessLayout = create_pipeline_layout(window, postProcessDescriptorSetLayouts);
	lut::PipelineLayout overVisWriteLayout = create_pipeline_layout(window, overVisWriteDescriptorSetLayouts);
	lut::PipelineLayout overVisReadLayout = create_pipeline_layout(window, overVisReadDescriptorSetLayouts);
	lut::PipelineLayout gBufWriteLayout = create_pipeline_layout(window, sceneDescriptorSetLayouts);
	lut::PipelineLayout deferredShadingLayout = create_pipeline_layout(window, deferredShadingDescriptorSetLayouts);
	lut::PipelineLayout shadowOffscreenLayout = create_pipeline_layout(window, shadowOffscreenDescriptorSetLayouts);

	PipelineLayouts pipelineLayouts{};
	pipelineLayouts.regularPipelineLayout = pipeLayout.handle;
	pipelineLayouts.postProcessPipelineLayout = postProcessLayout.handle;
	pipelineLayouts.overVisWritePipelineLayout = overVisWriteLayout.handle;
	pipelineLayouts.overVisReadPipelineLayout = overVisReadLayout.handle;
	pipelineLayouts.gBufWritePipelineLayout = gBufWriteLayout.handle;
	pipelineLayouts.deferredShadingPipelineLayout = deferredShadingLayout.handle;
	pipelineLayouts.shadowOffscreenPipelineLayout = shadowOffscreenLayout.handle;

	// Create pipelines
	lut::Pipeline pipeline = create_pipeline(window, renderPass.handle, pipeLayout.handle);
	lut::Pipeline alphaPipeline = create_alpha_pipeline(window, renderPass.handle, pipeLayout.handle);
	lut::Pipeline alphaOffscreenPipeline = create_alpha_pipeline(window, offscreenRenderPass.handle, pipeLayout.handle);
	lut::Pipeline debugPipeline = create_debug_pipeline(window, renderPass.handle, debugPipeLayout.handle);
	lut::Pipeline offscreePipeline = create_pipeline(window, offscreenRenderPass.handle, pipeLayout.handle);
	lut::Pipeline postProcessPipeline = create_post_process_pipeline(window, postProcessRenderPass.handle, postProcessLayout.handle);
	auto [overVisWritePipe, overVisReadPipe] = create_over_visualisations_pipeline(window, overVisualisationsRenderPass.handle, overVisWriteLayout.handle, overVisReadLayout.handle);
	auto [gBufWritePipe, deferredShadingPipe] = create_deferred_shading_pipeline(window, deferredShadingRenderPass.handle, gBufWriteLayout.handle, deferredShadingLayout.handle);
	lut::Pipeline shadowOffscreenPipeline = create_shadow_pipeline(window, shadowOffscreenRenderPass.handle, shadowOffscreenLayout.handle);

	Pipelines pipelines{};
	pipelines.regularPipeline = pipeline.handle;
	pipelines.alphaPipeline = alphaPipeline.handle;
	pipelines.alphaOffscreenPipeline = alphaOffscreenPipeline.handle;
	pipelines.debugPipeline = debugPipeline.handle;
	pipelines.offscreenPipeline = offscreePipeline.handle;
	pipelines.postProcessPipeline = postProcessPipeline.handle;
	pipelines.overVisWritePipeline = overVisWritePipe.handle;
	pipelines.overVisReadPipeline = overVisReadPipe.handle;
	pipelines.gBufWritePipline = gBufWritePipe.handle;
	pipelines.deferredShadingPipeline = deferredShadingPipe.handle;
	pipelines.shadowOffscreenPipeline = shadowOffscreenPipeline.handle;

	// Create depth specific depth buffer
	auto [DdepthBuffer, DdepthBufferView] = create_depth_buffer(window, allocator, VK_IMAGE_ASPECT_DEPTH_BIT);
	// Create stencil specific depth buffer
	auto [SdepthBuffer, SdepthBufferView] = create_depth_buffer(window, allocator, VK_IMAGE_ASPECT_STENCIL_BIT);
	// Create colour image buffer for post processing
	auto [colourBuffer, colourBufferView] = create_colour_buffer(window, allocator);
	// Create normals buffer
	auto [normalsBuffer, normalsBufferView] = create_normals_buffer(window, allocator);
	// Create tex coords buffer
	auto [albedoBuffer, albedoBufferView] = create_albedo_buffer(window, allocator);
	// Create shadow depth buffer
	auto [shadowDepthBuffer, shadowDepthBufferView] = create_shadow_depth_buffer(window, allocator);

	// Create offscreen framebuffer 
	lut::Framebuffer offscreenFramebuffer = create_offscreen_framebuffer(window, offscreenRenderPass.handle, colourBufferView.handle, DdepthBufferView.handle);
	// Create swapchain framebuffers
	std::vector<lut::Framebuffer> regularFramebuffers;
	create_regular_swapchain_framebuffers(window, renderPass.handle, regularFramebuffers, DdepthBufferView.handle);
	// Create framebuffers for doing post processing
	std::vector<lut::Framebuffer> fullscreenFramebuffers;
	create_fullscreen_swapchain_framebuffers(window, postProcessRenderPass.handle, fullscreenFramebuffers);
	// Create framebuffers for over visualisation debug
	std::vector<lut::Framebuffer> overVisulisationFramebuffers;
	create_over_visualisation_framebuffers(window, overVisualisationsRenderPass.handle, overVisulisationFramebuffers, colourBufferView.handle, SdepthBufferView.handle);
	// Create framebuffers for deferred shading
	std::vector<lut::Framebuffer> deferredShadingFramebuffers;
	create_deferred_shading_framebuffers(window, deferredShadingRenderPass.handle, deferredShadingFramebuffers, DdepthBufferView.handle, normalsBufferView.handle, albedoBufferView.handle);
	// Create shadow offscreen framebuffers
	std::vector<lut::Framebuffer> shadowFramebuffers;
	create_shadow_offscreen_framebuffers(window, shadowOffscreenRenderPass.handle, shadowFramebuffers, shadowDepthBufferView.handle);

	Framebuffers aFramebuffers{};
	aFramebuffers.offscreenFramebuffer = offscreenFramebuffer.handle;

	// Create command pool
	lut::CommandPool cpool = lut::create_command_pool(window, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

	// Setup synchronisation
	std::size_t frameIndex = 0;
	std::vector<VkCommandBuffer> cbuffers;
	std::vector<lut::Fence> frameDone;
	std::vector<lut::Semaphore> imageAvailable, renderFinished;

	for (std::size_t i = 0; i < regularFramebuffers.size(); ++i) {
		cbuffers.emplace_back(lut::alloc_command_buffer(window, cpool.handle));
		frameDone.emplace_back(lut::create_fence(window, VK_FENCE_CREATE_SIGNALED_BIT));
		imageAvailable.emplace_back(lut::create_semaphore(window));
		renderFinished.emplace_back(lut::create_semaphore(window));
	}

#pragma region UniformBuffers

	// Create Uniform Buffer
	lut::Buffer sceneUBO = lut::create_buffer(
		allocator, 
		sizeof(glsl::SceneUniform),
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		0,
		VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
	);

	// Create Light Buffer
	lut::Buffer lightUBO = lut::create_buffer(
		allocator,
		sizeof(glsl::LightUniform),
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		0,
		VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
	);

	// Create Debug Uniform Buffer
	lut::Buffer debugUBO = lut::create_buffer(
		allocator,
		sizeof(glsl::DebugUniform),
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		0,
		VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
	);

	// Create Lights Uniform Buffer
	lut::Buffer multipleLightsUBO = lut::create_buffer(
		allocator,
		sizeof(glsl::MultipleLights),
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		0,
		VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
	);

	// Create Depth MVP Buffer
	lut::Buffer depthMVPUBO = lut::create_buffer(
		allocator,
		sizeof(glsl::DepthMVP),
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		0,
		VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
	);

	UBOs ubos{};
	ubos.sceneUBO = sceneUBO.buffer;
	ubos.lightUBO = lightUBO.buffer;
	ubos.debugUBO = debugUBO.buffer;
	ubos.multipleLightsUBO = multipleLightsUBO.buffer;
	ubos.depthMVPUBO = depthMVPUBO.buffer;

#pragma endregion

	// Create descriptor pool
	lut::DescriptorPool dpool = lut::create_descriptor_pool(window);

	// Create sampler
	lut::Sampler sampler = lut::create_default_sampler(window);
	lut::Sampler shadowSampler = lut::create_shadow_sampler(window);

#pragma region DescriptorSets

	// Initialise scene descriptor set
	VkDescriptorSet sceneDescriptors = lut::alloc_desc_set(window, dpool.handle, sceneLayout.handle);
	{
		VkWriteDescriptorSet desc[1]{};

		VkDescriptorBufferInfo sceneUboInfo{};
		sceneUboInfo.buffer = sceneUBO.buffer;
		sceneUboInfo.range = VK_WHOLE_SIZE;

		desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[0].dstSet = sceneDescriptors;
		desc[0].dstBinding = 0;
		desc[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		desc[0].descriptorCount = 1;
		desc[0].pBufferInfo = &sceneUboInfo;

		constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
		vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
	}

	// Create light descriptor set
	VkDescriptorSet lightDescriptor = lut::alloc_desc_set(window, dpool.handle, uboLayout.handle);
	{
		VkWriteDescriptorSet desc[1]{};

		VkDescriptorBufferInfo lightUboInfo{};
		lightUboInfo.buffer = lightUBO.buffer;
		lightUboInfo.range = VK_WHOLE_SIZE;

		desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[0].dstSet = lightDescriptor;
		desc[0].dstBinding = 0;
		desc[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		desc[0].descriptorCount = 1;
		desc[0].pBufferInfo = &lightUboInfo;

		constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
		vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
	}

	// Create debug descriptor set
	VkDescriptorSet debugDescriptor = lut::alloc_desc_set(window, dpool.handle, uboLayout.handle);
	{
		VkWriteDescriptorSet desc[1]{};

		VkDescriptorBufferInfo debugUboInfo{};
		debugUboInfo.buffer = debugUBO.buffer;
		debugUboInfo.range = VK_WHOLE_SIZE;

		desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[0].dstSet = debugDescriptor;
		desc[0].dstBinding = 0;
		desc[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		desc[0].descriptorCount = 1;
		desc[0].pBufferInfo = &debugUboInfo;

		constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
		vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
	}

	VkDescriptorSet postProcessDescriptor = lut::alloc_desc_set(window, dpool.handle, postProcessDescriptorLayout.handle);
	{
		VkWriteDescriptorSet desc[1]{};

		VkDescriptorImageInfo outputColorInfo{};
		outputColorInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		outputColorInfo.imageView = colourBufferView.handle;
		outputColorInfo.sampler = sampler.handle;

		desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[0].dstSet = postProcessDescriptor;
		desc[0].dstBinding = 0;
		desc[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		desc[0].descriptorCount = 1;
		desc[0].pImageInfo = &outputColorInfo;

		constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
		vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
	}

	// Create over visualisation descriptor
	VkDescriptorSet overVisualisationDescriptor = lut::alloc_desc_set(window, dpool.handle, overVisualisationDescriptorLayout.handle);
	{
		VkWriteDescriptorSet desc[1]{};

		VkDescriptorImageInfo inputAttachments[1]{};
		inputAttachments[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		inputAttachments[0].imageView = SdepthBufferView.handle;
		inputAttachments[0].sampler = VK_NULL_HANDLE;

		desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[0].dstSet = overVisualisationDescriptor;
		desc[0].dstBinding = 0;
		desc[0].descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
		desc[0].descriptorCount = 1;
		desc[0].pImageInfo = &inputAttachments[0];

		constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
		vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
	}

	// Create deferred shading descriptor
	VkDescriptorSet deferredShadingDescriptor = lut::alloc_desc_set(window, dpool.handle, deferredShadingDescriptorLayout.handle);
	{
		VkWriteDescriptorSet desc[3]{};

		VkDescriptorImageInfo inputAttachments[3]{};
		inputAttachments[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		inputAttachments[0].imageView = normalsBufferView.handle;
		inputAttachments[0].sampler = VK_NULL_HANDLE;

		inputAttachments[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		inputAttachments[1].imageView = albedoBufferView.handle;
		inputAttachments[1].sampler = VK_NULL_HANDLE;

		inputAttachments[2].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		inputAttachments[2].imageView = DdepthBufferView.handle;
		inputAttachments[2].sampler = VK_NULL_HANDLE; 

		desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[0].dstSet = deferredShadingDescriptor;
		desc[0].dstBinding = 0;
		desc[0].descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
		desc[0].descriptorCount = 1;
		desc[0].pImageInfo = &inputAttachments[0];

		desc[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[1].dstSet = deferredShadingDescriptor;
		desc[1].dstBinding = 1;
		desc[1].descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
		desc[1].descriptorCount = 1;
		desc[1].pImageInfo = &inputAttachments[1];

		desc[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[2].dstSet = deferredShadingDescriptor;
		desc[2].dstBinding = 2;
		desc[2].descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
		desc[2].descriptorCount = 1;
		desc[2].pImageInfo = &inputAttachments[2];
		
		constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
		vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
	}

	// Create lights descriptor set
	VkDescriptorSet mutlipleLightsDescriptor = lut::alloc_desc_set(window, dpool.handle, uboLayout.handle);
	{
		VkWriteDescriptorSet desc[1]{};

		VkDescriptorBufferInfo lightsUboInfo{};
		lightsUboInfo.buffer = multipleLightsUBO.buffer;
		lightsUboInfo.range = VK_WHOLE_SIZE;

		desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[0].dstSet = mutlipleLightsDescriptor;
		desc[0].dstBinding = 0;
		desc[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		desc[0].descriptorCount = 1;
		desc[0].pBufferInfo = &lightsUboInfo;

		constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
		vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
	}

	// Create depth MVP descriptor set
	VkDescriptorSet depthMVPDescriptor = lut::alloc_desc_set(window, dpool.handle, uboLayoutVert.handle);
	{
		VkWriteDescriptorSet desc[1]{};

		VkDescriptorBufferInfo depthUboInfo{};
		depthUboInfo.buffer = depthMVPUBO.buffer;
		depthUboInfo.range = VK_WHOLE_SIZE;

		desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[0].dstSet = depthMVPDescriptor;
		desc[0].dstBinding = 0;
		desc[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		desc[0].descriptorCount = 1;
		desc[0].pBufferInfo = &depthUboInfo;

		constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
		vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
	}

	// Create shadow map image descriptor set
	VkDescriptorSet shadowMapDescriptor = lut::alloc_desc_set(window, dpool.handle, fragImageLayout.handle);
	{
		VkWriteDescriptorSet desc[1]{};

		VkDescriptorImageInfo imageInfo[1]{};
		imageInfo[0].imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
		imageInfo[0].imageView = shadowDepthBufferView.handle;
		imageInfo[0].sampler = shadowSampler.handle;

		desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[0].dstSet = shadowMapDescriptor;
		desc[0].dstBinding = 0;
		desc[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		desc[0].descriptorCount = 1;
		desc[0].pImageInfo = imageInfo;

		constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
		vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
	}

#pragma endregion

	// Load mesh data
	// Load baked model
	BakedModel bakedModel = load_baked_model(cfg::kModelPath);
	
	// Load all texture images and image views
	// std::vector<lut::Image> textures;
	std::vector<lut::ImageView> textureViews;
	for (BakedTextureInfo textureInfo : bakedModel.textures) {
		lut::CommandPool texCmdPool = lut::create_command_pool(window, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);
		textureViews.push_back(load_mesh_texture(window, texCmdPool.handle, allocator, textureInfo));
	}

#pragma region MaterialDescriptorSets

	// Vector of dummy texture image views outside of the loop so they don't get destructed
	// when control goes out of loop
	std::vector<lut::ImageView> dummyTextures;

	// Create Descriptor Sets for each material
	std::vector<VkDescriptorSet> materialDescriptors;
	for (std::size_t i = 0; i < bakedModel.materials.size(); i++) {
		VkDescriptorSet materialDescriptor = lut::alloc_desc_set(window, dpool.handle, materialLayout.handle);
		VkWriteDescriptorSet desc[5]{};

		VkDescriptorImageInfo baseColourInfo{};
		baseColourInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		baseColourInfo.imageView = textureViews[bakedModel.materials[i].baseColorTextureId].handle;
		baseColourInfo.sampler = sampler.handle;

		desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[0].dstSet = materialDescriptor;
		desc[0].dstBinding = 0;
		desc[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		desc[0].descriptorCount = 1;
		desc[0].pImageInfo = &baseColourInfo;

		VkDescriptorImageInfo metalnessInfo{};
		metalnessInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		metalnessInfo.imageView = textureViews[bakedModel.materials[i].metalnessTextureId].handle;
		metalnessInfo.sampler = sampler.handle;

		desc[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[1].dstSet = materialDescriptor;
		desc[1].dstBinding = 1;
		desc[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		desc[1].descriptorCount = 1;
		desc[1].pImageInfo = &metalnessInfo;

		VkDescriptorImageInfo roughnessInfo{};
		roughnessInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		roughnessInfo.imageView = textureViews[bakedModel.materials[i].roughnessTextureId].handle;
		roughnessInfo.sampler = sampler.handle;

		desc[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[2].dstSet = materialDescriptor;
		desc[2].dstBinding = 2;
		desc[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		desc[2].descriptorCount = 1;
		desc[2].pImageInfo = &roughnessInfo;

		// Check if the material has a valid alphaMaskTextureId, otherwise set its
		// imageView handle to the base colour texture
		VkImageView alphaMaskImageView = VK_NULL_HANDLE;
		if (bakedModel.materials[i].alphaMaskTextureId == 0xffffffff) {
			alphaMaskImageView = textureViews[bakedModel.materials[i].baseColorTextureId].handle;
		} else {
			alphaMaskImageView = textureViews[bakedModel.materials[i].alphaMaskTextureId].handle;
		}

		VkDescriptorImageInfo alphaMaskInfo{};
		alphaMaskInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		alphaMaskInfo.imageView = alphaMaskImageView;
		alphaMaskInfo.sampler = sampler.handle;

		desc[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[3].dstSet = materialDescriptor;
		desc[3].dstBinding = 3;
		desc[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		desc[3].descriptorCount = 1;
		desc[3].pImageInfo = &alphaMaskInfo;

		VkDescriptorImageInfo normalMapInfo{};
		normalMapInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		normalMapInfo.imageView = textureViews[bakedModel.materials[i].normalMapTextureId].handle;
		normalMapInfo.sampler = sampler.handle;

		desc[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[4].dstSet = materialDescriptor;
		desc[4].dstBinding = 4;
		desc[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		desc[4].descriptorCount = 1;
		desc[4].pImageInfo = &normalMapInfo;
			
		constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
		vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
		materialDescriptors.emplace_back(materialDescriptor);
	}

	//for (std::size_t i = 0; i < bakedModel.materials.size(); i++) {
	//	std::cout << "i: " << i << "textureViews[" << bakedModel.materials[i].normalMapTextureId << "]: " << textureViews[bakedModel.materials[i].normalMapTextureId].handle << std::endl;
	//}

#pragma endregion

	DescriptorSets descriptorSets{
		materialDescriptors,
		sceneDescriptors,
		lightDescriptor,
		debugDescriptor,
		postProcessDescriptor,
		overVisualisationDescriptor,
		deferredShadingDescriptor,
		mutlipleLightsDescriptor,
		depthMVPDescriptor,
		shadowMapDescriptor
	};

#pragma region MeshData

	// Mesh Data
	std::vector<MeshData> meshData;
	for (std::size_t i = 0; i < bakedModel.meshes.size(); i++) {
		lut::Buffer vertexPosGPU = lut::create_buffer(
			allocator,
			bakedModel.meshes[i].positions.size() * sizeof(glm::vec3),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			0,
			VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
		);

		lut::Buffer vertexTexGPU = lut::create_buffer(
			allocator,
			bakedModel.meshes[i].texcoords.size() * sizeof(glm::vec2),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			0,
			VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
		);

		// Even though we pass the TBN frame, this is kept in to allow me to compare and get the
		// screenshots to compare against the TBN normal mapping
		lut::Buffer vertexNormGPU = lut::create_buffer(
			allocator,
			bakedModel.meshes[i].normals.size() * sizeof(glm::vec3),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			0,
			VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
		);

		lut::Buffer vertexTangGPU = lut::create_buffer(
			allocator,
			bakedModel.meshes[i].tangentsComp.size() * sizeof(std::uint32_t),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			0,
			VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
		);

		lut::Buffer vertexIndexGPU = lut::create_buffer(
			allocator,
			bakedModel.meshes[i].indices.size() * sizeof(std::uint32_t),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			0,
			VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
		);

		lut::Buffer posStaging = lut::create_buffer(
			allocator,
			bakedModel.meshes[i].positions.size() * sizeof(glm::vec3),
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
		);

		lut::Buffer texStaging = lut::create_buffer(
			allocator,
			bakedModel.meshes[i].texcoords.size() * sizeof(glm::vec2),
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
		);

		lut::Buffer normStaging = lut::create_buffer(
			allocator,
			bakedModel.meshes[i].normals.size() * sizeof(glm::vec3),
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
		);

		lut::Buffer tangStaging = lut::create_buffer(
			allocator,
			bakedModel.meshes[i].tangentsComp.size() * sizeof(std::uint32_t),
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
		);

		lut::Buffer indexStaging = lut::create_buffer(
			allocator,
			bakedModel.meshes[i].indices.size() * sizeof(std::uint32_t),
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
		);

		void* posPtr = nullptr;
		if (const auto res = vmaMapMemory(allocator.allocator, posStaging.allocation, &posPtr); VK_SUCCESS != res)
			throw lut::Error("Mapping memory for writing\n vmaMapMemory() returned %s", lut::to_string(res).c_str());

		std::memcpy(posPtr, bakedModel.meshes[i].positions.data(), bakedModel.meshes[i].positions.size() * sizeof(glm::vec3));
		vmaUnmapMemory(allocator.allocator, posStaging.allocation);

		void* texPtr = nullptr;
		if (const auto res = vmaMapMemory(allocator.allocator, texStaging.allocation, &texPtr); VK_SUCCESS != res)
			throw lut::Error("Mapping memory for writing\n vmaMapMemory() returned %s", lut::to_string(res).c_str());

		std::memcpy(texPtr, bakedModel.meshes[i].texcoords.data(), bakedModel.meshes[i].texcoords.size() * sizeof(glm::vec2));
		vmaUnmapMemory(allocator.allocator, texStaging.allocation);

		void* normPtr = nullptr;
		if (const auto res = vmaMapMemory(allocator.allocator, normStaging.allocation, &normPtr); VK_SUCCESS != res)
			throw lut::Error("Mapping memory for writing\n vmaMapMemory() returned %s", lut::to_string(res).c_str());

		std::memcpy(normPtr, bakedModel.meshes[i].normals.data(), bakedModel.meshes[i].normals.size() * sizeof(glm::vec3));
		vmaUnmapMemory(allocator.allocator, normStaging.allocation);

		void* tangPtr = nullptr;
		if (const auto res = vmaMapMemory(allocator.allocator, tangStaging.allocation, &tangPtr); VK_SUCCESS != res)
			throw lut::Error("Mapping memory for writing\n vmaMapMemory() returned %s", lut::to_string(res).c_str());

		std::memcpy(tangPtr, bakedModel.meshes[i].tangentsComp.data(), bakedModel.meshes[i].tangentsComp.size() * sizeof(std::uint32_t));
		vmaUnmapMemory(allocator.allocator, tangStaging.allocation);

		void* indexPtr = nullptr;
		if (const auto res = vmaMapMemory(allocator.allocator, indexStaging.allocation, &indexPtr); VK_SUCCESS != res)
			throw lut::Error("Mapping memory for writing\n vmaMapMemory() returned %s", lut::to_string(res).c_str());

		std::memcpy(indexPtr, bakedModel.meshes[i].indices.data(), bakedModel.meshes[i].indices.size() * sizeof(std::uint32_t));
		vmaUnmapMemory(allocator.allocator, indexStaging.allocation);

		lut::Fence uploadComplete = create_fence(window);

		lut::CommandPool uploadPool = create_command_pool(window);
		VkCommandBuffer uploadCmd = alloc_command_buffer(window, uploadPool.handle);
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = 0;
		beginInfo.pInheritanceInfo = nullptr;

		if (const auto res = vkBeginCommandBuffer(uploadCmd, &beginInfo); VK_SUCCESS != res)
		throw lut::Error("Unable to begin command buffer\n vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str());

		VkBufferCopy pcopy{};
		pcopy.size = bakedModel.meshes[i].positions.size() * sizeof(glm::vec3);

		vkCmdCopyBuffer(uploadCmd, posStaging.buffer, vertexPosGPU.buffer, 1, &pcopy);

		lut::buffer_barrier(
			uploadCmd,
			vertexPosGPU.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
		);

		VkBufferCopy tcopy{};
		tcopy.size = bakedModel.meshes[i].texcoords.size() * sizeof(glm::vec2);

		vkCmdCopyBuffer(uploadCmd, texStaging.buffer, vertexTexGPU.buffer, 1, &tcopy);

		lut::buffer_barrier(
			uploadCmd,
			vertexTexGPU.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
		);

		VkBufferCopy ncopy{};
		ncopy.size = bakedModel.meshes[i].normals.size() * sizeof(glm::vec3);

		vkCmdCopyBuffer(uploadCmd, normStaging.buffer, vertexNormGPU.buffer, 1, &ncopy);

		lut::buffer_barrier(
			uploadCmd,
			vertexNormGPU.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
		);

		VkBufferCopy tangcopy{};
		tangcopy.size = bakedModel.meshes[i].tangentsComp.size() * sizeof(std::uint32_t);

		vkCmdCopyBuffer(uploadCmd, tangStaging.buffer, vertexTangGPU.buffer, 1, &tangcopy);

		lut::buffer_barrier(
			uploadCmd,
			vertexTangGPU.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
		);

		VkBufferCopy icopy{};
		icopy.size = bakedModel.meshes[i].indices.size() * sizeof(std::uint32_t);

		vkCmdCopyBuffer(uploadCmd, indexStaging.buffer, vertexIndexGPU.buffer, 1, &icopy);

		lut::buffer_barrier(
			uploadCmd,
			vertexIndexGPU.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
		);

		if (const auto res = vkEndCommandBuffer(uploadCmd); VK_SUCCESS != res)
		throw lut::Error("Unable to end command buffer\n vkEndCommandBuffer() returned %s", lut::to_string(res).c_str());

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &uploadCmd;

		if (const auto res =  vkQueueSubmit(window.graphicsQueue, 1, &submitInfo, uploadComplete.handle); VK_SUCCESS != res)
			throw lut::Error("Unable to submit commands\n vkQueueSubmit() returned %s", lut::to_string(res).c_str());

		if (const auto res = vkWaitForFences(window.device, 1, &uploadComplete.handle, VK_TRUE, std::numeric_limits<std::uint64_t>::max()) ; VK_SUCCESS != res)
			throw lut::Error("Unable to wait for fences\n vkWaitForFences() returned %s", lut::to_string(res).c_str());

		bool hasAlphaMask = false;
		if (bakedModel.materials[bakedModel.meshes[i].materialId].alphaMaskTextureId != 0xffffffff) hasAlphaMask = true;

		meshData.emplace_back(
			MeshData {
				std::move(vertexPosGPU), 
				std::move(vertexTexGPU),
				std::move(vertexNormGPU), // Even though we pass the TBN frame, this is kept in to allow me to compare and get the
										  // screenshots to compare against the TBN normal mapping
				std::move(vertexTangGPU),
				std::move(vertexIndexGPU), 
				bakedModel.meshes[i].indices.size(),
				bakedModel.meshes[i].materialId,
				hasAlphaMask
			});
	}

#pragma endregion

	// Application main loop
	bool recreateSwapchain = false;

	auto previousClock = Clock_::now();
	while (!glfwWindowShouldClose(window.window)) {
		glfwPollEvents();

		if (recreateSwapchain) {
			vkDeviceWaitIdle(window.device);
		
			const auto changes = recreate_swapchain(window);

			if (changes.changedFormat) {
				renderPass = create_render_pass(window);
				offscreenRenderPass = create_offscreen_render_pass(window);
				postProcessRenderPass = create_post_process_render_pass(window);
				overVisualisationsRenderPass = create_over_visualisations_render_pass(window);
				deferredShadingRenderPass = create_deferred_shading_render_pass(window);
				shadowOffscreenRenderPass = create_offscreen_shadow_render_pass(window);

				renderPasses.regularRenderPass = renderPass.handle;
				renderPasses.offscreenRenderPass = offscreenRenderPass.handle;
				renderPasses.postProcessRenderPass = postProcessRenderPass.handle;
				renderPasses.overVisualisationsRenderPass = overVisualisationsRenderPass.handle;
				renderPasses.deferredShadingRenderPass = deferredShadingRenderPass.handle;
				renderPasses.shadowOffscreenRenderPass = shadowOffscreenRenderPass.handle;
			}
				
			if (changes.changedSize) {
				std::tie(colourBuffer, colourBufferView) = create_colour_buffer(window, allocator);
				std::tie(DdepthBuffer, DdepthBufferView) = create_depth_buffer(window, allocator, VK_IMAGE_ASPECT_DEPTH_BIT);
				std::tie(SdepthBuffer, SdepthBufferView) = create_depth_buffer(window, allocator, VK_IMAGE_ASPECT_STENCIL_BIT);
				std::tie(normalsBuffer, normalsBufferView) = create_normals_buffer(window, allocator);
				std::tie(albedoBuffer, albedoBufferView) = create_albedo_buffer(window, allocator);
			}
			
			offscreenFramebuffer = create_offscreen_framebuffer(window, offscreenRenderPass.handle, colourBufferView.handle, DdepthBufferView.handle);
			aFramebuffers.offscreenFramebuffer = offscreenFramebuffer.handle; 
			regularFramebuffers.clear();
			create_regular_swapchain_framebuffers(window, renderPass.handle, regularFramebuffers, DdepthBufferView.handle);
			fullscreenFramebuffers.clear();
			create_fullscreen_swapchain_framebuffers(window, postProcessRenderPass.handle, fullscreenFramebuffers);
			overVisulisationFramebuffers.clear();
			create_over_visualisation_framebuffers(window, overVisualisationsRenderPass.handle, overVisulisationFramebuffers, colourBufferView.handle, SdepthBufferView.handle);
			deferredShadingFramebuffers.clear();
			create_deferred_shading_framebuffers(window, deferredShadingRenderPass.handle, deferredShadingFramebuffers, DdepthBufferView.handle, normalsBufferView.handle, albedoBufferView.handle);
			shadowFramebuffers.clear();
			create_shadow_offscreen_framebuffers(window, shadowOffscreenRenderPass.handle, shadowFramebuffers, shadowDepthBufferView.handle);

			if (changes.changedSize) {
				pipeline = create_pipeline(window, renderPass.handle, pipeLayout.handle);
				alphaPipeline = create_alpha_pipeline(window, renderPass.handle, pipeLayout.handle);
				alphaOffscreenPipeline = create_alpha_pipeline(window, offscreenRenderPass.handle, pipeLayout.handle);
				debugPipeline = create_debug_pipeline(window, renderPass.handle, debugPipeLayout.handle);
				offscreePipeline = create_pipeline(window, offscreenRenderPass.handle, pipeLayout.handle);
				postProcessPipeline = create_post_process_pipeline(window, postProcessRenderPass.handle, postProcessLayout.handle);
				std::tie(overVisWritePipe, overVisReadPipe) = create_over_visualisations_pipeline(window, overVisualisationsRenderPass.handle, overVisWriteLayout.handle, overVisReadLayout.handle);
				std::tie(gBufWritePipe, deferredShadingPipe) = create_deferred_shading_pipeline(window, deferredShadingRenderPass.handle, gBufWriteLayout.handle, deferredShadingLayout.handle);

				pipelines.regularPipeline = pipeline.handle;
				pipelines.alphaPipeline = alphaPipeline.handle;
				pipelines.alphaOffscreenPipeline = alphaOffscreenPipeline.handle;
				pipelines.debugPipeline = debugPipeline.handle;
				pipelines.offscreenPipeline = offscreePipeline.handle;
				pipelines.postProcessPipeline = postProcessPipeline.handle;
				pipelines.overVisWritePipeline = overVisWritePipe.handle;
				pipelines.overVisReadPipeline = overVisReadPipe.handle;
				pipelines.gBufWritePipline = gBufWritePipe.handle;
				pipelines.deferredShadingPipeline = deferredShadingPipe.handle;
			}

			// Recreate post process descriptor since it relies on colour buffer
			postProcessDescriptor = lut::alloc_desc_set(window, dpool.handle, postProcessDescriptorLayout.handle);
			{
				VkWriteDescriptorSet desc[1]{};

				VkDescriptorImageInfo outputColorInfo{};
				outputColorInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				outputColorInfo.imageView = colourBufferView.handle;
				outputColorInfo.sampler = sampler.handle;

				desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				desc[0].dstSet = postProcessDescriptor;
				desc[0].dstBinding = 0;
				desc[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				desc[0].descriptorCount = 1;
				desc[0].pImageInfo = &outputColorInfo;

				constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
				vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
			}

			// Recreate over visualisation descriptor since it relies on colour buffer
			overVisualisationDescriptor = lut::alloc_desc_set(window, dpool.handle, overVisualisationDescriptorLayout.handle);
			{
				VkWriteDescriptorSet desc[1]{};

				VkDescriptorImageInfo inputAttachments[1]{};
				inputAttachments[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				inputAttachments[0].imageView = SdepthBufferView.handle;
				inputAttachments[0].sampler = VK_NULL_HANDLE;

				desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				desc[0].dstSet = overVisualisationDescriptor;
				desc[0].dstBinding = 0;
				desc[0].descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
				desc[0].descriptorCount = 1;
				desc[0].pImageInfo = &inputAttachments[0];

				constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
				vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
			}

			// Recreate deferred shading descriptor as it relies on normals and albedo buffers
			deferredShadingDescriptor = lut::alloc_desc_set(window, dpool.handle, deferredShadingDescriptorLayout.handle);
			{
				VkWriteDescriptorSet desc[3]{};

				VkDescriptorImageInfo inputAttachments[3]{};
				inputAttachments[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				inputAttachments[0].imageView = normalsBufferView.handle;
				inputAttachments[0].sampler = VK_NULL_HANDLE;

				inputAttachments[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				inputAttachments[1].imageView = albedoBufferView.handle;
				inputAttachments[1].sampler = VK_NULL_HANDLE;

				inputAttachments[2].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				inputAttachments[2].imageView = DdepthBufferView.handle;
				inputAttachments[2].sampler = VK_NULL_HANDLE;

				desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				desc[0].dstSet = deferredShadingDescriptor;
				desc[0].dstBinding = 0;
				desc[0].descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
				desc[0].descriptorCount = 1;
				desc[0].pImageInfo = &inputAttachments[0];

				desc[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				desc[1].dstSet = deferredShadingDescriptor;
				desc[1].dstBinding = 1;
				desc[1].descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
				desc[1].descriptorCount = 1;
				desc[1].pImageInfo = &inputAttachments[1];

				desc[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				desc[2].dstSet = deferredShadingDescriptor;
				desc[2].dstBinding = 2;
				desc[2].descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
				desc[2].descriptorCount = 1;
				desc[2].pImageInfo = &inputAttachments[2];

				constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
				vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
			}

			descriptorSets.postProcessDescriptor = postProcessDescriptor;
			descriptorSets.overVisualisationDescriptor = overVisualisationDescriptor;
			descriptorSets.deferredShadingDescriptor = deferredShadingDescriptor;

			recreateSwapchain = false;
			continue;
		}

		frameIndex++;
		frameIndex %= cbuffers.size();

		assert(frameIndex < frameDone.size());

		if (const auto res = vkWaitForFences(window.device, 1, &frameDone[frameIndex].handle, VK_TRUE, std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res)
			throw lut::Error("Unable to wait for frame fence %u\n vkWaitForFences() returned %s", frameIndex, lut::to_string(res).c_str());

		assert(frameIndex < imageAvailable.size());

		std::uint32_t imageIndex = 0;
		const auto acquireRes = vkAcquireNextImageKHR(
			window.device,
			window.swapchain,
			std::numeric_limits<std::uint64_t>::max(),
			imageAvailable[frameIndex].handle,
			VK_NULL_HANDLE,
			&imageIndex
		);

		if (VK_SUBOPTIMAL_KHR == acquireRes || VK_ERROR_OUT_OF_DATE_KHR == acquireRes) {
			recreateSwapchain = true;

			--frameIndex;
			frameIndex %= cbuffers.size();

			continue;
		}

		if (VK_SUCCESS != acquireRes)
			throw lut::Error("Unable to acquire next swapchain image\n vkAcquireNextImageKHR() returned %s", lut::to_string(acquireRes).c_str());

		if (const auto res = vkResetFences(window.device, 1, &frameDone[frameIndex].handle); VK_SUCCESS != res)
			throw lut::Error("Unable to reset frame fence %u\n vkResetFences() returned %s", frameIndex, lut::to_string(res).c_str());
	
		const auto now = Clock_::now();
		const auto dt = std::chrono::duration_cast<Secondsf_>(now - previousClock).count();
		previousClock = now;

		update_user_state(state, dt); 

		assert(std::size_t(frameIndex) < cbuffers.size());
		assert(std::size_t(imageIndex) < regularFramebuffers.size());

		glsl::SceneUniform sceneUniforms{};
		glsl::DebugUniform debugUniforms{};
		glsl::DepthMVP depthMVPUniform{};
		update_scene_uniforms(sceneUniforms, window.swapchainExtent.width, window.swapchainExtent.height, state);
		update_debug_uniforms(debugUniforms, state);

		glsl::LightUniform lightUniforms{};
		lightUniforms.lightPos = glm::vec4(-0.2972f, 7.3100f, -11.9532f, 0.0f);
		lightUniforms.lightColour = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);

		update_depth_mvp_uniforms(depthMVPUniform, window.swapchainExtent.width, window.swapchainExtent.height, lightUniforms.lightPos);

		glsl::MultipleLights multipleLightsUniform{};
		// The 8 braziers in the main room surrounding statue
		multipleLightsUniform.light[0].lightPos = glm::vec4(8.0377f, -1.1000f, -15.8845f, 0.0f);
		multipleLightsUniform.light[1].lightPos = glm::vec4(5.6632f, -1.1000f, -12.2500f, 0.0f);
		multipleLightsUniform.light[2].lightPos = glm::vec4(2.1255f, -1.1000f, -10.1000f, 0.0f);
		multipleLightsUniform.light[3].lightPos = glm::vec4(-2.0867f, -1.1000f, -10.2500f, 0.0f);
		multipleLightsUniform.light[4].lightPos = glm::vec4(-5.7367f, -1.1000f, -12.2000f, 0.0f);
		multipleLightsUniform.light[5].lightPos = glm::vec4(-7.8244f, -1.1000f, -15.8500f, 0.0f);
		multipleLightsUniform.light[6].lightPos = glm::vec4(3.0632f, -1.1000f, -25.8500f, 0.0f);
		multipleLightsUniform.light[7].lightPos = glm::vec4(-3.0632f, -1.1000f, -25.8500f, 0.0f);
		// The 1 lone brazier in the smaller room with statue
		multipleLightsUniform.light[8].lightPos = glm::vec4(-7.4867f, -1.1000f, -36.0500f, 0.0f);
		// 3 next to statue in smaller room
		multipleLightsUniform.light[9].lightPos = glm::vec4(-2.4367f, -3.1000f, -46.5500f, 0.0f);
		multipleLightsUniform.light[10].lightPos = glm::vec4(2.3632f, -3.1000f, -46.5500f, 0.0f);
		multipleLightsUniform.light[11].lightPos = glm::vec4(-0.0514f, -3.9000f, -49.6000f, 0.0f);
		// 2 going into back hallway
		multipleLightsUniform.light[12].lightPos = glm::vec4(-7.2367f, -3.1000f, -61.2500f, 0.0f);
		multipleLightsUniform.light[13].lightPos = glm::vec4(7.1088f, -3.1000f, -61.2500f, 0.0f);
		// 4 before large middle one
		multipleLightsUniform.light[14].lightPos = glm::vec4(-2.0867f, -3.1000f, -66.5500f, 0.0f);
		multipleLightsUniform.light[15].lightPos = glm::vec4(2.0867f, -3.1000f, -66.5500f, 0.0f);
		multipleLightsUniform.light[16].lightPos = glm::vec4(-2.0867f, -3.1000f, -68.9000f, 0.0f);
		multipleLightsUniform.light[17].lightPos = glm::vec4(2.0867f, -3.1000f, -68.9000f, 0.0f);
		// 1 large middle one
		multipleLightsUniform.light[18].lightPos = glm::vec4(0.0000f, -4.1000f, -76.0000f, 0.0f);
		// 2 final back ones
		multipleLightsUniform.light[19].lightPos = glm::vec4(-1.6867f, -3.1000f, -88.8500f, 0.0f);
		multipleLightsUniform.light[20].lightPos = glm::vec4(1.6867f, -3.1000f, -88.8500f, 0.0f);

		// Give all lights a 'fire' orange colour
		for (int i = 0; i < 21; i++) {
			multipleLightsUniform.light[i].lightColour = glm::vec4(0.9882f, 0.4549f, 0.0196f, 1.0f);
		}		

		Uniforms uniforms{};
		uniforms.sceneUniforms = sceneUniforms;
		uniforms.debugUniforms = debugUniforms;
		uniforms.lightUniforms = lightUniforms;
		uniforms.multipleLightsUniform = multipleLightsUniform;
		uniforms.depthMVPUniform = depthMVPUniform;

		aFramebuffers.regularSwapchainFramebuffer = regularFramebuffers[imageIndex].handle;
		aFramebuffers.fullscreenSwapchainFramebuffer = fullscreenFramebuffers[imageIndex].handle;
		aFramebuffers.overVisualisationFramebuffer = overVisulisationFramebuffers[imageIndex].handle;
		aFramebuffers.deferredShadingFramebuffer = deferredShadingFramebuffers[imageIndex].handle;
		aFramebuffers.shadowOffscreenFramebuffer = shadowFramebuffers[imageIndex].handle;

		record_commands(
			cbuffers[frameIndex],
			renderPasses,
			aFramebuffers,
			pipelines,
			window.swapchainExtent,
			meshData,
			ubos,
			uniforms,
			pipelineLayouts,
			descriptorSets,
			state
		);

		assert(std::size_t(frameIndex) < renderFinished.size());
		
		submit_commands(
			window,
			cbuffers[frameIndex],
			frameDone[frameIndex].handle,
			imageAvailable[frameIndex].handle,
			renderFinished[frameIndex].handle
		);

		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = &renderFinished[frameIndex].handle;
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = &window.swapchain;
		presentInfo.pImageIndices = &imageIndex;
		presentInfo.pResults = nullptr;

		const auto presentRes = vkQueuePresentKHR(window.presentQueue, &presentInfo);

		if (VK_SUBOPTIMAL_KHR == presentRes || VK_ERROR_OUT_OF_DATE_KHR == presentRes)
			recreateSwapchain = true;
		else if (VK_SUCCESS != presentRes)
			throw lut::Error("Unable to present swapchain image %u\n vkQueuePresentKHR() returned %s", imageIndex, lut::to_string(presentRes).c_str());
	}

	vkDeviceWaitIdle(window.device);
	images.clear();

	return 0;
}
catch( std::exception const& eErr )
{
	std::fprintf( stderr, "\n" );
	std::fprintf( stderr, "Error: %s\n", eErr.what() );
	return 1;
}

// Anonymous namespace
namespace {
	void glfw_callback_key_press( GLFWwindow* aWindow, int aKey, int /*aScanCode*/, int aAction, int /*aModifierFlags*/ )
	{
		if( GLFW_KEY_ESCAPE == aKey && GLFW_PRESS == aAction )
		{
			glfwSetWindowShouldClose( aWindow, GLFW_TRUE );
		}

		auto state = static_cast<UserState*>(glfwGetWindowUserPointer(aWindow));
		assert(state);

		bool const isReleased = (GLFW_RELEASE == aAction);

		if (isReleased) {
			switch(aKey) {
				case GLFW_KEY_1:
					// Regular visualisation
					state->debugVisualisation = 1;
					//printf("Debug: 1\n");
					break;
				case GLFW_KEY_2:
					// Utilisation of texture mipmap levels
					state->debugVisualisation = 2;
					//printf("Debug: 2\n");
					break;
				case GLFW_KEY_3:
					// Fragment depth
					state->debugVisualisation = 3;
					//printf("Debug: 3\n");
					break;
				case GLFW_KEY_4:
					// Partial derivatives of the per-fragment depth
					state->debugVisualisation = 4;
					//printf("Debug: 4\n");
					break;
				case GLFW_KEY_5:
					// Toggle mosaic effect
					state->mosaicEffect = !state->mosaicEffect;
					break;
				case GLFW_KEY_6:
					// Overdraw visualisation
					state->debugVisualisation = 5;
					break;
				case GLFW_KEY_7:
					// Overshading visualisation
					state->debugVisualisation = 6;
					break;
				case GLFW_KEY_8:
					// Toggle deferred shading
					state->deferredShading = !state->deferredShading;
					break;
				case GLFW_KEY_9:
					// Toggle shadows
					state->shadows = !state->shadows;
					break;
				default:
				;
			}
		}

		switch(aKey) {
			case GLFW_KEY_W:
				state->inputMap[std::size_t(EInputState::forward)] = !isReleased;
				break;
			case GLFW_KEY_S:
				state->inputMap[std::size_t(EInputState::backward)] = !isReleased;
				break;
			case GLFW_KEY_A:
				state->inputMap[std::size_t(EInputState::strafeLeft)] = !isReleased;
				break;
			case GLFW_KEY_D:
				state->inputMap[std::size_t(EInputState::strafeRight)] = !isReleased;
				break;
			case GLFW_KEY_E:
				state->inputMap[std::size_t(EInputState::levitate)] = !isReleased;
				break;
			case GLFW_KEY_Q:
				state->inputMap[std::size_t(EInputState::sink)] = !isReleased;
				break;

			case GLFW_KEY_LEFT_SHIFT: [[fallthrough]];
			case GLFW_KEY_RIGHT_SHIFT:
				state->inputMap[std::size_t(EInputState::fast)] = !isReleased;
				break;

			case GLFW_KEY_LEFT_CONTROL: [[fallthrough]];
			case GLFW_KEY_RIGHT_CONTROL:
				state->inputMap[std::size_t(EInputState::slow)] = !isReleased;
				break;

			default:
			;
		}
	}

	void glfw_callback_button(GLFWwindow* aWindow, int aButton, int aAction, int aMods) {
		auto state = static_cast<UserState*>(glfwGetWindowUserPointer(aWindow));
		assert(state);

		if (GLFW_MOUSE_BUTTON_RIGHT == aButton && GLFW_PRESS == aAction) {
			auto& flag = state->inputMap[std::size_t(EInputState::mousing)];

			flag = !flag;
			if (flag)
				glfwSetInputMode(aWindow, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
			else
				glfwSetInputMode(aWindow, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		}
	}

	void glfw_callback_motion(GLFWwindow* aWindow, double x, double y) {
			auto state = static_cast<UserState*>(glfwGetWindowUserPointer(aWindow));
			assert(state);

			state->mouseX = float(x);
			state->mouseY = float(y);
	}

	lut::RenderPass create_render_pass(const lut::VulkanWindow& aWindow) {
		VkAttachmentDescription attachments[2]{};
		attachments[0].format = aWindow.swapchainFormat;
		attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		attachments[1].format = cfg::kDepthFormat;
		attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;

		VkAttachmentReference subpassAttachments[1]{};
		subpassAttachments[0].attachment = 0;
		subpassAttachments[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference depthAttachment{};
		depthAttachment.attachment = 1;
		depthAttachment.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpasses[1]{};
		subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpasses[0].colorAttachmentCount = 1;
		subpasses[0].pColorAttachments = subpassAttachments;
		subpasses[0].pDepthStencilAttachment = &depthAttachment;

		VkSubpassDependency deps[2]{};
		deps[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
		deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
		deps[0].srcAccessMask = 0;
		deps[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		deps[0].dstSubpass = 0;
		deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		deps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

		deps[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
		deps[1].srcSubpass = VK_SUBPASS_EXTERNAL;
		deps[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		deps[1].srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
		deps[1].dstSubpass = 0;
		deps[1].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
		deps[1].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;

		VkRenderPassCreateInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		passInfo.attachmentCount = 2;
		passInfo.pAttachments = attachments;
		passInfo.subpassCount = 1;
		passInfo.pSubpasses = subpasses;
		passInfo.dependencyCount = 2;
		passInfo.pDependencies = deps;
	
		VkRenderPass rpass = VK_NULL_HANDLE;
		if (const auto res = vkCreateRenderPass(aWindow.device, &passInfo, nullptr, &rpass); VK_SUCCESS != res) {
			throw lut::Error("Unable to create render pass\n vkCreateRenderPass() returned %s\n", lut::to_string(res).c_str());
		}

		return lut::RenderPass(aWindow.device, rpass);
	}

	// Draws to intermediate texture image for post processing
	lut::RenderPass create_offscreen_render_pass(const lut::VulkanWindow& aWindow) {
		VkAttachmentDescription attachments[2]{};
		attachments[0].format = VK_FORMAT_R8G8B8A8_SRGB;
		attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		attachments[1].format = cfg::kDepthFormat;
		attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;

		VkAttachmentReference subpassAttachments[1]{};
		subpassAttachments[0].attachment = 0;
		subpassAttachments[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference depthAttachment{};
		depthAttachment.attachment = 1;
		depthAttachment.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpasses[1]{};
		subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpasses[0].colorAttachmentCount = 1;
		subpasses[0].pColorAttachments = subpassAttachments;
		subpasses[0].pDepthStencilAttachment = &depthAttachment;

		VkSubpassDependency deps[3]{};
		deps[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
		deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
		deps[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		deps[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
		deps[0].dstSubpass = 0;
		deps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

		deps[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
		deps[1].srcSubpass = VK_SUBPASS_EXTERNAL;
		deps[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		deps[1].srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
		deps[1].dstSubpass = 0;
		deps[1].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
		deps[1].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;

		deps[2].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
		deps[2].srcSubpass = 0;
		deps[2].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		deps[2].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		deps[2].dstSubpass = VK_SUBPASS_EXTERNAL;
		deps[2].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		deps[2].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

		VkRenderPassCreateInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		passInfo.attachmentCount = 2;
		passInfo.pAttachments = attachments;
		passInfo.subpassCount = 1;
		passInfo.pSubpasses = subpasses;
		passInfo.dependencyCount = 3;
		passInfo.pDependencies = deps;
	
		VkRenderPass rpass = VK_NULL_HANDLE;
		if (const auto res = vkCreateRenderPass(aWindow.device, &passInfo, nullptr, &rpass); VK_SUCCESS != res) {
			throw lut::Error("Unable to create render pass\n vkCreateRenderPass() returned %s\n", lut::to_string(res).c_str());
		}

		return lut::RenderPass(aWindow.device, rpass);
	}

	lut::RenderPass create_post_process_render_pass(const lut::VulkanWindow& aWindow) {
		VkAttachmentDescription attachments[1]{};
		attachments[0].format = aWindow.swapchainFormat;
		attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentReference subpassAttachments[1]{};
		subpassAttachments[0].attachment = 0;
		subpassAttachments[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpasses[1]{};
		subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpasses[0].colorAttachmentCount = 1;
		subpasses[0].pColorAttachments = subpassAttachments;

		VkSubpassDependency deps[1]{};
		deps[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
		deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
		deps[0].srcAccessMask = 0;
		deps[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		deps[0].dstSubpass = 0;
		deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		deps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

		VkRenderPassCreateInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		passInfo.attachmentCount = 1;
		passInfo.pAttachments = attachments;
		passInfo.subpassCount = 1;
		passInfo.pSubpasses = subpasses;
		passInfo.dependencyCount = 1;
		passInfo.pDependencies = deps;
	
		VkRenderPass rpass = VK_NULL_HANDLE;
		if (const auto res = vkCreateRenderPass(aWindow.device, &passInfo, nullptr, &rpass); VK_SUCCESS != res) {
			throw lut::Error("Unable to create render pass\n vkCreateRenderPass() returned %s\n", lut::to_string(res).c_str());
		}

		return lut::RenderPass(aWindow.device, rpass);
	}

	lut::RenderPass create_over_visualisations_render_pass(const lut::VulkanWindow& aWindow) {
		// First subpass writes to depth/stencil buffer
		// Second subpass uses depth/stencil buffer to write to swapchain image

		VkAttachmentDescription attachments[3]{};
		// Swap chain image attachment
		attachments[0].format = aWindow.swapchainFormat;
		attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
		attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

		attachments[1].format = VK_FORMAT_R8G8B8A8_SRGB;
		attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[1].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

		attachments[2].format = cfg::kDepthFormat;
		attachments[2].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[2].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[2].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachments[2].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[2].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		attachments[2].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[2].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

		VkAttachmentReference swapchainAttachment{};
		swapchainAttachment.attachment = 0;
		swapchainAttachment.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference colorAttachment{};
		colorAttachment.attachment = 1;
		colorAttachment.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference stencilAttachment{};
		stencilAttachment.attachment = 2;
		stencilAttachment.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpasses[2]{};
		subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpasses[0].colorAttachmentCount = 1;
		subpasses[0].pColorAttachments = &colorAttachment;
		subpasses[0].pDepthStencilAttachment = &stencilAttachment;

		subpasses[1].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpasses[1].colorAttachmentCount = 1;
		subpasses[1].pColorAttachments = &swapchainAttachment;

		VkAttachmentReference inputAttachments[2]{};
		inputAttachments[0].attachment = 1;
		inputAttachments[0].layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		inputAttachments[1].attachment = 2;
		inputAttachments[1].layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		subpasses[1].inputAttachmentCount = 2;
		subpasses[1].pInputAttachments = inputAttachments;

		VkSubpassDependency deps[4]{};
		deps[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
		deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
		deps[0].srcAccessMask = 0;
		deps[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		deps[0].dstSubpass = 0;
		deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		deps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;

		deps[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
		deps[1].srcSubpass = VK_SUBPASS_EXTERNAL;
		deps[1].srcAccessMask = 0;
		deps[1].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		deps[1].dstSubpass = 1;
		deps[1].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		deps[1].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

		deps[2].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
		deps[2].srcSubpass = 0;
		deps[2].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
		deps[2].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
		deps[2].dstSubpass = 1;
		deps[2].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_INPUT_ATTACHMENT_READ_BIT;
		deps[2].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;

		deps[3].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
		deps[3].srcSubpass = 1;
		deps[3].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		deps[3].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
		deps[3].dstSubpass = VK_SUBPASS_EXTERNAL;
		deps[3].dstAccessMask = 0;
		deps[3].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;

		VkRenderPassCreateInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		passInfo.attachmentCount = 3;
		passInfo.pAttachments = attachments;
		passInfo.subpassCount = 2;
		passInfo.pSubpasses = subpasses;
		passInfo.dependencyCount = 4;
		passInfo.pDependencies = deps;

		VkRenderPass rpass = VK_NULL_HANDLE;
		if (const auto res = vkCreateRenderPass(aWindow.device, &passInfo, nullptr, &rpass); VK_SUCCESS != res) {
			throw lut::Error("Unable to create render pass\n vkCreateRenderPass() returned %s\n", lut::to_string(res).c_str());
		}

		return lut::RenderPass(aWindow.device, rpass);
	}

	lut::RenderPass create_deferred_shading_render_pass(const lut::VulkanWindow& aWindow) {
		// Geometry pass to write data to G-Buffers then 2nd pass
		// for shading

		VkAttachmentDescription attachments[4]{};
		// Swap chain image attachment
		attachments[0].format = aWindow.swapchainFormat;
		attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
		attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

		// Normal Attachment
		attachments[1].format = VK_FORMAT_R16G16B16A16_SFLOAT; // Maybe R10G10B10A2?
		attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[1].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

		// Albedo Attachment
		attachments[2].format = VK_FORMAT_R16G16B16A16_SFLOAT; // Maybe reduce to 16 bit components since 32 may not be needed
		attachments[2].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[2].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[2].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachments[2].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[2].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		attachments[2].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[2].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

		// Depth Attachment
		attachments[3].format = cfg::kDepthFormat;
		attachments[3].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[3].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[3].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[3].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[3].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		attachments[3].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[3].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

		VkAttachmentReference swapchainAttachment{};
		swapchainAttachment.attachment = 0;
		swapchainAttachment.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference depthAttachment{};
		depthAttachment.attachment = 3;
		depthAttachment.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkAttachmentReference normalAndAlbedoAttachments[2]{};
		normalAndAlbedoAttachments[0].attachment = 1;
		normalAndAlbedoAttachments[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		normalAndAlbedoAttachments[1].attachment = 2;
		normalAndAlbedoAttachments[1].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpasses[2]{};
		subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpasses[0].colorAttachmentCount = 2;
		subpasses[0].pColorAttachments = normalAndAlbedoAttachments;
		subpasses[0].pDepthStencilAttachment = &depthAttachment;

		subpasses[1].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpasses[1].colorAttachmentCount = 1;
		subpasses[1].pColorAttachments = &swapchainAttachment;

		VkAttachmentReference inputAttachments[3]{};
		inputAttachments[0].attachment = 1;
		inputAttachments[0].layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		inputAttachments[1].attachment = 2;
		inputAttachments[1].layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		inputAttachments[2].attachment = 3;
		inputAttachments[2].layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		subpasses[1].inputAttachmentCount = 3;
		subpasses[1].pInputAttachments = inputAttachments;

		VkSubpassDependency deps[4]{};
		deps[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
		deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
		deps[0].srcAccessMask = 0;
		deps[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		deps[0].dstSubpass = 0;
		deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		deps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;

		deps[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
		deps[1].srcSubpass = VK_SUBPASS_EXTERNAL;
		deps[1].srcAccessMask = 0;
		deps[1].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		deps[1].dstSubpass = 1;
		deps[1].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		deps[1].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

		deps[2].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
		deps[2].srcSubpass = 0;
		deps[2].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
		deps[2].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
		deps[2].dstSubpass = 1;
		deps[2].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_INPUT_ATTACHMENT_READ_BIT;
		deps[2].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;

		deps[3].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
		deps[3].srcSubpass = 1;
		deps[3].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		deps[3].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
		deps[3].dstSubpass = VK_SUBPASS_EXTERNAL;
		deps[3].dstAccessMask = 0;
		deps[3].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;

		VkRenderPassCreateInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		passInfo.attachmentCount = 4;
		passInfo.pAttachments = attachments;
		passInfo.subpassCount = 2;
		passInfo.pSubpasses = subpasses;
		passInfo.dependencyCount = 4;
		passInfo.pDependencies = deps;

		VkRenderPass rpass = VK_NULL_HANDLE;
		if (const auto res = vkCreateRenderPass(aWindow.device, &passInfo, nullptr, &rpass); VK_SUCCESS != res) {
			throw lut::Error("Unable to create render pass\n vkCreateRenderPass() returned %s\n", lut::to_string(res).c_str());
		}

		return lut::RenderPass(aWindow.device, rpass);
	}

	lut::RenderPass create_offscreen_shadow_render_pass(const lut::VulkanWindow& aWindow) {
		VkAttachmentDescription attachments[1]{};
		attachments[0].format = cfg::kDepthFormat;
		attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[0].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
		attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;

		VkAttachmentReference depthAttachment{};
		depthAttachment.attachment = 0;
		depthAttachment.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpasses[1]{};
		subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpasses[0].colorAttachmentCount = 0;
		subpasses[0].pDepthStencilAttachment = &depthAttachment;

		VkSubpassDependency deps[2]{};
		deps[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
		deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
		deps[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
		deps[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		deps[0].dstSubpass = 0;
		deps[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		deps[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;

		deps[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
		deps[1].srcSubpass = 0;
		deps[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		deps[1].srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
		deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
		deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		deps[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

		VkRenderPassCreateInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		passInfo.attachmentCount = 1;
		passInfo.pAttachments = attachments;
		passInfo.subpassCount = 1;
		passInfo.pSubpasses = subpasses;
		passInfo.dependencyCount = 2;
		passInfo.pDependencies = deps;

		VkRenderPass rpass = VK_NULL_HANDLE;
		if (const auto res = vkCreateRenderPass(aWindow.device, &passInfo, nullptr, &rpass); VK_SUCCESS != res) {
			throw lut::Error("Unable to create render pass\n vkCreateRenderPass() returned %s\n", lut::to_string(res).c_str());
		}

		return lut::RenderPass(aWindow.device, rpass);
	}

	lut::DescriptorSetLayout create_scene_descriptor_layout(const lut::VulkanWindow& aWindow) {
		VkDescriptorSetLayoutBinding bindings[1]{};
		bindings[0].binding = 0;
		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		bindings[0].descriptorCount = 1;
		bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
		layoutInfo.pBindings = bindings;

		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (const auto res = vkCreateDescriptorSetLayout(aWindow.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
			throw lut::Error("Unable to create descriptor set layout\n vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str());

		return lut::DescriptorSetLayout(aWindow.device, layout);
	}

	lut::DescriptorSetLayout create_material_descriptor_layout(const lut::VulkanWindow& aWindow) {
		VkDescriptorSetLayoutBinding bindings[5]{};
		bindings[0].binding = 0;
		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		bindings[0].descriptorCount = 1;
		bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		bindings[1].binding = 1;
		bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		bindings[1].descriptorCount = 1;
		bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		bindings[2].binding = 2;
		bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		bindings[2].descriptorCount = 1;
		bindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		bindings[3].binding = 3;
		bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		bindings[3].descriptorCount = 1;
		bindings[3].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		bindings[4].binding = 4;
		bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		bindings[4].descriptorCount = 1;
		bindings[4].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
		layoutInfo.pBindings = bindings;

		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (const auto res = vkCreateDescriptorSetLayout(aWindow.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
			throw lut::Error("Unable to create descriptor set layout\n vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str());

		return lut::DescriptorSetLayout(aWindow.device, layout);
	}

	lut::DescriptorSetLayout create_fragment_ubo_descriptor_layout(const lut::VulkanWindow& aWindow) {
		VkDescriptorSetLayoutBinding bindings[1]{};
		bindings[0].binding = 0;
		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		bindings[0].descriptorCount = 1;
		bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
		layoutInfo.pBindings = bindings;

		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (const auto res = vkCreateDescriptorSetLayout(aWindow.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
			throw lut::Error("Unable to create descriptor set layout\n vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str());

		return lut::DescriptorSetLayout(aWindow.device, layout);
	}

	lut::DescriptorSetLayout create_vertex_ubo_descriptor_layout(const lut::VulkanWindow& aWindow) {
		VkDescriptorSetLayoutBinding bindings[1]{};
		bindings[0].binding = 0;
		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		bindings[0].descriptorCount = 1;
		bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
		layoutInfo.pBindings = bindings;

		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (const auto res = vkCreateDescriptorSetLayout(aWindow.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
			throw lut::Error("Unable to create descriptor set layout\n vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str());

		return lut::DescriptorSetLayout(aWindow.device, layout);
	}

	lut::DescriptorSetLayout create_post_process_descriptor_layout(const lut::VulkanWindow& aWindow) {
		VkDescriptorSetLayoutBinding bindings[1]{};
		bindings[0].binding = 0;
		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		bindings[0].descriptorCount = 1;
		bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
		layoutInfo.pBindings = bindings;

		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (const auto res = vkCreateDescriptorSetLayout(aWindow.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
			throw lut::Error("Unable to create descriptor set layout\n vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str());

		return lut::DescriptorSetLayout(aWindow.device, layout);
	}

	lut::DescriptorSetLayout create_over_visualisations_descriptor_layout(const lut::VulkanWindow& aWindow) {
		VkDescriptorSetLayoutBinding bindings[1]{};
		bindings[0].binding = 0;
		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
		bindings[0].descriptorCount = 1;
		bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
		layoutInfo.pBindings = bindings;

		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (const auto res = vkCreateDescriptorSetLayout(aWindow.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
			throw lut::Error("Unable to create descriptor set layout\n vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str());

		return lut::DescriptorSetLayout(aWindow.device, layout);

	}

	lut::DescriptorSetLayout create_deferred_shading_descriptor_layout(const lut::VulkanWindow& aWindow) {
		VkDescriptorSetLayoutBinding bindings[3]{};
		bindings[0].binding = 0;
		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
		bindings[0].descriptorCount = 1;
		bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		bindings[1].binding = 1;
		bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
		bindings[1].descriptorCount = 1;
		bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		bindings[2].binding = 2;
		bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
		bindings[2].descriptorCount = 1;
		bindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
		layoutInfo.pBindings = bindings;

		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (const auto res = vkCreateDescriptorSetLayout(aWindow.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
			throw lut::Error("Unable to create descriptor set layout\n vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str());

		return lut::DescriptorSetLayout(aWindow.device, layout);
	}

	lut::DescriptorSetLayout create_fragment_image_layout(const lut::VulkanWindow& aWindow) {
		VkDescriptorSetLayoutBinding bindings[1]{};
		bindings[0].binding = 0;
		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		bindings[0].descriptorCount = 1;
		bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
		layoutInfo.pBindings = bindings;

		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (const auto res = vkCreateDescriptorSetLayout(aWindow.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
			throw lut::Error("Unable to create descriptor set layout\n vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str());

		return lut::DescriptorSetLayout(aWindow.device, layout);
	}

	lut::PipelineLayout create_pipeline_layout(const lut::VulkanWindow& aWindow, std::vector<VkDescriptorSetLayout>& aDescriptorSetLayouts) {
		VkPipelineLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		layoutInfo.setLayoutCount = aDescriptorSetLayouts.size();
		layoutInfo.pSetLayouts = aDescriptorSetLayouts.data();
		layoutInfo.pushConstantRangeCount = 0;
		layoutInfo.pPushConstantRanges = nullptr;

		VkPipelineLayout layout = VK_NULL_HANDLE;
		if (const auto res = vkCreatePipelineLayout(aWindow.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res) {
			throw lut::Error("Unable to create pipeline layout\n vkCreatePipelineLayout() returned %s", lut::to_string(res).c_str());
		}

		return lut::PipelineLayout(aWindow.device, layout);
	}

	lut::Pipeline create_pipeline(const lut::VulkanWindow& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout) {
		lut::ShaderModule vert = lut::load_shader_module(aWindow, cfg::kVertShaderPath);
		lut::ShaderModule frag = lut::load_shader_module(aWindow, cfg::kFragShaderPath);

		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vert.handle;
		stages[0].pName = "main";

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = frag.handle;
		stages[1].pName = "main";

		// Even though we pass the TBN frame, normals are kept in to allow me to compare and get the
		// screenshots to compare against the TBN normal mapping for the report
		VkVertexInputBindingDescription vertexInputs[4]{};
		vertexInputs[0].binding = 0;
		vertexInputs[0].stride = sizeof(float) * 3;
		vertexInputs[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		vertexInputs[1].binding = 1;
		vertexInputs[1].stride = sizeof(float) * 2;
		vertexInputs[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		vertexInputs[2].binding = 2;
		vertexInputs[2].stride = sizeof(float) * 3;
		vertexInputs[2].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		vertexInputs[3].binding = 3;
		vertexInputs[3].stride = sizeof(std::uint32_t) * 1;
		vertexInputs[3].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		VkVertexInputAttributeDescription vertexAttributes[4]{};
		vertexAttributes[0].binding = 0;
		vertexAttributes[0].location = 0;
		vertexAttributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[0].offset = 0;
		vertexAttributes[1].binding = 1;
		vertexAttributes[1].location = 1;
		vertexAttributes[1].format = VK_FORMAT_R32G32_SFLOAT;
		vertexAttributes[1].offset = 0;
		vertexAttributes[2].binding = 2;
		vertexAttributes[2].location = 2;
		vertexAttributes[2].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[2].offset = 0;
		vertexAttributes[3].binding = 3;
		vertexAttributes[3].location = 3;
		vertexAttributes[3].format = VK_FORMAT_A2R10G10B10_UNORM_PACK32;
		vertexAttributes[3].offset = 0;

		VkPipelineVertexInputStateCreateInfo inputInfo{};
		inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		inputInfo.vertexBindingDescriptionCount = 4;
		inputInfo.pVertexBindingDescriptions = vertexInputs;
		inputInfo.vertexAttributeDescriptionCount = 4;
		inputInfo.pVertexAttributeDescriptions = vertexAttributes;

		VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
		assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyInfo.primitiveRestartEnable = VK_FALSE;

		VkViewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = float(aWindow.swapchainExtent.width);
		viewport.height = float(aWindow.swapchainExtent.height);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		VkRect2D scissor{};
		scissor.offset = VkOffset2D{0, 0};
		scissor.extent = aWindow.swapchainExtent;

		VkPipelineViewportStateCreateInfo viewportInfo{};
		viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewport;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = &scissor;

		VkPipelineRasterizationStateCreateInfo rasterInfo{};
		rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterInfo.depthClampEnable = VK_FALSE;
		rasterInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterInfo.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterInfo.depthBiasEnable = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterInfo.lineWidth = 1.0f;

		VkPipelineMultisampleStateCreateInfo samplingInfo{};
		samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		VkPipelineColorBlendAttachmentState blendStates[1]{};
		blendStates[0].blendEnable = VK_FALSE;
		blendStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

		VkPipelineColorBlendStateCreateInfo blendInfo{};
		blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blendInfo.logicOpEnable = VK_FALSE;
		blendInfo.attachmentCount = 1;
		blendInfo.pAttachments = blendStates;

		VkPipelineDepthStencilStateCreateInfo depthInfo{};
		depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthInfo.depthTestEnable = VK_TRUE;
		depthInfo.depthWriteEnable = VK_TRUE;
		depthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthInfo.minDepthBounds = 0.0f;
		depthInfo.maxDepthBounds = 1.0f;

		VkGraphicsPipelineCreateInfo pipeInfo{};
		pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipeInfo.stageCount = 2;
		pipeInfo.pStages = stages;
		pipeInfo.pVertexInputState = &inputInfo;
		pipeInfo.pInputAssemblyState = &assemblyInfo;
		pipeInfo.pTessellationState = nullptr;
		pipeInfo.pViewportState = &viewportInfo;
		pipeInfo.pRasterizationState = &rasterInfo;
		pipeInfo.pMultisampleState = &samplingInfo;
		pipeInfo.pDepthStencilState = &depthInfo;
		pipeInfo.pColorBlendState = &blendInfo;
		pipeInfo.pDynamicState = nullptr;
		pipeInfo.layout = aPipelineLayout;
		pipeInfo.renderPass = aRenderPass;
		pipeInfo.subpass = 0;

		VkPipeline pipe = VK_NULL_HANDLE;
		if (const auto res = vkCreateGraphicsPipelines(aWindow.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipe); VK_SUCCESS != res) {
			throw lut::Error("Unable to create graphics pipeline\n vkCreateGraphicsPipeline() returned %s", lut::to_string(res).c_str());
		}

		return lut::Pipeline(aWindow.device, pipe);
	}

	lut::Pipeline create_alpha_pipeline( lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout )
	{
		lut::ShaderModule vert = lut::load_shader_module(aWindow, cfg::kVertShaderPath);
		lut::ShaderModule frag = lut::load_shader_module(aWindow, cfg::kFragShaderPath);

		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vert.handle;
		stages[0].pName = "main";

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = frag.handle;
		stages[1].pName = "main";

		// Even though we pass the TBN frame, normals are kept in to allow me to compare and get the
		// screenshots to compare against the TBN normal mapping for the report
		VkVertexInputBindingDescription vertexInputs[4]{};
		vertexInputs[0].binding = 0;
		vertexInputs[0].stride = sizeof(float) * 3;
		vertexInputs[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		vertexInputs[1].binding = 1;
		vertexInputs[1].stride = sizeof(float) * 2;
		vertexInputs[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		vertexInputs[2].binding = 2;
		vertexInputs[2].stride = sizeof(float) * 3;
		vertexInputs[2].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		vertexInputs[3].binding = 3;
		vertexInputs[3].stride = sizeof(std::uint32_t) * 1;
		vertexInputs[3].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		VkVertexInputAttributeDescription vertexAttributes[4]{};
		vertexAttributes[0].binding = 0;
		vertexAttributes[0].location = 0;
		vertexAttributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[0].offset = 0;
		vertexAttributes[1].binding = 1;
		vertexAttributes[1].location = 1;
		vertexAttributes[1].format = VK_FORMAT_R32G32_SFLOAT;
		vertexAttributes[1].offset = 0;
		vertexAttributes[2].binding = 2;
		vertexAttributes[2].location = 2;
		vertexAttributes[2].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[2].offset = 0;
		vertexAttributes[3].binding = 3;
		vertexAttributes[3].location = 3;
		vertexAttributes[3].format = VK_FORMAT_A2R10G10B10_UNORM_PACK32;
		vertexAttributes[3].offset = 0;

		VkPipelineVertexInputStateCreateInfo inputInfo{};
		inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		inputInfo.vertexBindingDescriptionCount = 4;
		inputInfo.pVertexBindingDescriptions = vertexInputs;
		inputInfo.vertexAttributeDescriptionCount = 4;
		inputInfo.pVertexAttributeDescriptions = vertexAttributes;

		VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
		assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyInfo.primitiveRestartEnable = VK_FALSE;

		VkViewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = float(aWindow.swapchainExtent.width);
		viewport.height = float(aWindow.swapchainExtent.height);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		VkRect2D scissor{};
		scissor.offset = VkOffset2D{0, 0};
		scissor.extent = aWindow.swapchainExtent;

		VkPipelineViewportStateCreateInfo viewportInfo{};
		viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewport;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = &scissor;

		VkPipelineRasterizationStateCreateInfo rasterInfo{};
		rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterInfo.depthClampEnable = VK_FALSE;
		rasterInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterInfo.cullMode = VK_CULL_MODE_NONE;
		rasterInfo.depthBiasEnable = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterInfo.lineWidth = 1.0f;

		VkPipelineMultisampleStateCreateInfo samplingInfo{};
		samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		VkPipelineColorBlendAttachmentState blendStates[1]{};
		blendStates[0].blendEnable = VK_FALSE;
		blendStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

		VkPipelineColorBlendStateCreateInfo blendInfo{};
		blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blendInfo.logicOpEnable = VK_FALSE;
		blendInfo.attachmentCount = 1;
		blendInfo.pAttachments = blendStates;

		VkPipelineDepthStencilStateCreateInfo depthInfo{};
		depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthInfo.depthTestEnable = VK_TRUE;
		depthInfo.depthWriteEnable = VK_TRUE;
		depthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthInfo.minDepthBounds = 0.0f;
		depthInfo.maxDepthBounds = 1.0f;

		VkGraphicsPipelineCreateInfo pipeInfo{};
		pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipeInfo.stageCount = 2;
		pipeInfo.pStages = stages;
		pipeInfo.pVertexInputState = &inputInfo;
		pipeInfo.pInputAssemblyState = &assemblyInfo;
		pipeInfo.pTessellationState = nullptr;
		pipeInfo.pViewportState = &viewportInfo;
		pipeInfo.pRasterizationState = &rasterInfo;
		pipeInfo.pMultisampleState = &samplingInfo;
		pipeInfo.pDepthStencilState = &depthInfo;
		pipeInfo.pColorBlendState = &blendInfo;
		pipeInfo.pDynamicState = nullptr;
		pipeInfo.layout = aPipelineLayout;
		pipeInfo.renderPass = aRenderPass;
		pipeInfo.subpass = 0;

		VkPipeline pipe = VK_NULL_HANDLE;
		if (const auto res = vkCreateGraphicsPipelines(aWindow.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipe); VK_SUCCESS != res) {
			throw lut::Error("Unable to create graphics pipeline\n vkCreateGraphicsPipeline() returned %s", lut::to_string(res).c_str());
		}

		return lut::Pipeline(aWindow.device, pipe);
	}

	lut::Pipeline create_debug_pipeline(const lut::VulkanWindow& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout) {
		lut::ShaderModule vert = lut::load_shader_module(aWindow, cfg::kDebugVertShaderPath);
		lut::ShaderModule frag = lut::load_shader_module(aWindow, cfg::kDebugFragShaderPath);

		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vert.handle;
		stages[0].pName = "main";

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = frag.handle;
		stages[1].pName = "main";

		VkVertexInputBindingDescription vertexInputs[2]{};
		vertexInputs[0].binding = 0;
		vertexInputs[0].stride = sizeof(float) * 3;
		vertexInputs[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		vertexInputs[1].binding = 1;
		vertexInputs[1].stride = sizeof(float) * 2;
		vertexInputs[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		VkVertexInputAttributeDescription vertexAttributes[2]{};
		vertexAttributes[0].binding = 0;
		vertexAttributes[0].location = 0;
		vertexAttributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[0].offset = 0;
		vertexAttributes[1].binding = 1;
		vertexAttributes[1].location = 1;
		vertexAttributes[1].format = VK_FORMAT_R32G32_SFLOAT;
		vertexAttributes[1].offset = 0;

		VkPipelineVertexInputStateCreateInfo inputInfo{};
		inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		inputInfo.vertexBindingDescriptionCount = 2;
		inputInfo.pVertexBindingDescriptions = vertexInputs;
		inputInfo.vertexAttributeDescriptionCount = 2;
		inputInfo.pVertexAttributeDescriptions = vertexAttributes;

		VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
		assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyInfo.primitiveRestartEnable = VK_FALSE;

		VkViewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = float(aWindow.swapchainExtent.width);
		viewport.height = float(aWindow.swapchainExtent.height);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		VkRect2D scissor{};
		scissor.offset = VkOffset2D{0, 0};
		scissor.extent = aWindow.swapchainExtent;

		VkPipelineViewportStateCreateInfo viewportInfo{};
		viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewport;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = &scissor;

		VkPipelineRasterizationStateCreateInfo rasterInfo{};
		rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterInfo.depthClampEnable = VK_FALSE;
		rasterInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterInfo.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterInfo.depthBiasEnable = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterInfo.lineWidth = 1.0f;

		VkPipelineMultisampleStateCreateInfo samplingInfo{};
		samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		VkPipelineColorBlendAttachmentState blendStates[1]{};
		blendStates[0].blendEnable = VK_FALSE;
		blendStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

		VkPipelineColorBlendStateCreateInfo blendInfo{};
		blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blendInfo.logicOpEnable = VK_FALSE;
		blendInfo.attachmentCount = 1;
		blendInfo.pAttachments = blendStates;

		VkPipelineDepthStencilStateCreateInfo depthInfo{};
		depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthInfo.depthTestEnable = VK_TRUE;
		depthInfo.depthWriteEnable = VK_TRUE;
		depthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthInfo.minDepthBounds = 0.0f;
		depthInfo.maxDepthBounds = 1.0f;

		VkGraphicsPipelineCreateInfo pipeInfo{};
		pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipeInfo.stageCount = 2;
		pipeInfo.pStages = stages;
		pipeInfo.pVertexInputState = &inputInfo;
		pipeInfo.pInputAssemblyState = &assemblyInfo;
		pipeInfo.pTessellationState = nullptr;
		pipeInfo.pViewportState = &viewportInfo;
		pipeInfo.pRasterizationState = &rasterInfo;
		pipeInfo.pMultisampleState = &samplingInfo;
		pipeInfo.pDepthStencilState = &depthInfo;
		pipeInfo.pColorBlendState = &blendInfo;
		pipeInfo.pDynamicState = nullptr;
		pipeInfo.layout = aPipelineLayout;
		pipeInfo.renderPass = aRenderPass;
		pipeInfo.subpass = 0;

		VkPipeline pipe = VK_NULL_HANDLE;
		if (const auto res = vkCreateGraphicsPipelines(aWindow.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipe); VK_SUCCESS != res) {
			throw lut::Error("Unable to create graphics pipeline\n vkCreateGraphicsPipeline() returned %s", lut::to_string(res).c_str());
		}

		return lut::Pipeline(aWindow.device, pipe);
	}

	lut::Pipeline create_post_process_pipeline(const lut::VulkanWindow& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout) {
		lut::ShaderModule vert = lut::load_shader_module(aWindow, cfg::kPPVertShaderPath);
		lut::ShaderModule frag = lut::load_shader_module(aWindow, cfg::kPPFragShaderPath);

		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vert.handle;
		stages[0].pName = "main";

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = frag.handle;
		stages[1].pName = "main";

		VkPipelineVertexInputStateCreateInfo inputInfo{};
		inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

		VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
		assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyInfo.primitiveRestartEnable = VK_FALSE;

		VkViewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = float(aWindow.swapchainExtent.width);
		viewport.height = float(aWindow.swapchainExtent.height);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		VkRect2D scissor{};
		scissor.offset = VkOffset2D{0, 0};
		scissor.extent = aWindow.swapchainExtent;

		VkPipelineViewportStateCreateInfo viewportInfo{};
		viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewport;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = &scissor;

		VkPipelineRasterizationStateCreateInfo rasterInfo{};
		rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterInfo.depthClampEnable = VK_FALSE;
		rasterInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterInfo.cullMode = VK_CULL_MODE_FRONT_BIT;
		rasterInfo.depthBiasEnable = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterInfo.lineWidth = 1.0f;

		VkPipelineMultisampleStateCreateInfo samplingInfo{};
		samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		VkPipelineColorBlendAttachmentState blendStates[1]{};
		blendStates[0].blendEnable = VK_FALSE;
		blendStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

		VkPipelineColorBlendStateCreateInfo blendInfo{};
		blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blendInfo.logicOpEnable = VK_FALSE;
		blendInfo.attachmentCount = 1;
		blendInfo.pAttachments = blendStates;

		VkGraphicsPipelineCreateInfo pipeInfo{};
		pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipeInfo.stageCount = 2;
		pipeInfo.pStages = stages;
		pipeInfo.pVertexInputState = &inputInfo;
		pipeInfo.pInputAssemblyState = &assemblyInfo;
		pipeInfo.pTessellationState = nullptr;
		pipeInfo.pViewportState = &viewportInfo;
		pipeInfo.pRasterizationState = &rasterInfo;
		pipeInfo.pMultisampleState = &samplingInfo;
		pipeInfo.pDepthStencilState = nullptr;
		pipeInfo.pColorBlendState = &blendInfo;
		pipeInfo.pDynamicState = nullptr;
		pipeInfo.layout = aPipelineLayout;
		pipeInfo.renderPass = aRenderPass;
		pipeInfo.subpass = 0;

		VkPipeline pipe = VK_NULL_HANDLE;
		if (const auto res = vkCreateGraphicsPipelines(aWindow.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipe); VK_SUCCESS != res) {
			throw lut::Error("Unable to create graphics pipeline\n vkCreateGraphicsPipeline() returned %s", lut::to_string(res).c_str());
		}

		return lut::Pipeline(aWindow.device, pipe);
	}

	std::tuple<lut::Pipeline, lut::Pipeline> create_over_visualisations_pipeline(const lut::VulkanWindow& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineWriteLayout, VkPipelineLayout aPipelineReadLayout) {
		lut::ShaderModule vert = lut::load_shader_module(aWindow, cfg::kOverVisWriteVertShaderPath);
		lut::ShaderModule frag = lut::load_shader_module(aWindow, cfg::kOverVisWriteFragShaderPath);

		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vert.handle;
		stages[0].pName = "main";

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = frag.handle;
		stages[1].pName = "main";

		VkVertexInputBindingDescription vertexInputs[1]{};
		vertexInputs[0].binding = 0;
		vertexInputs[0].stride = sizeof(float) * 3;
		vertexInputs[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		VkVertexInputAttributeDescription vertexAttributes[1]{};
		vertexAttributes[0].binding = 0;
		vertexAttributes[0].location = 0;
		vertexAttributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[0].offset = 0;

		VkPipelineVertexInputStateCreateInfo inputInfo{};
		inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		inputInfo.vertexBindingDescriptionCount = 1;
		inputInfo.pVertexBindingDescriptions = vertexInputs;
		inputInfo.vertexAttributeDescriptionCount = 1;
		inputInfo.pVertexAttributeDescriptions = vertexAttributes;

		VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
		assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyInfo.primitiveRestartEnable = VK_FALSE;

		VkViewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = float(aWindow.swapchainExtent.width);
		viewport.height = float(aWindow.swapchainExtent.height);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		VkRect2D scissor{};
		scissor.offset = VkOffset2D{ 0, 0 };
		scissor.extent = aWindow.swapchainExtent;

		VkPipelineViewportStateCreateInfo viewportInfo{};
		viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewport;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = &scissor;

		VkPipelineRasterizationStateCreateInfo rasterInfo{};
		rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterInfo.depthClampEnable = VK_FALSE;
		rasterInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterInfo.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterInfo.depthBiasEnable = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterInfo.lineWidth = 1.0f;

		VkPipelineMultisampleStateCreateInfo samplingInfo{};
		samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		VkPipelineColorBlendAttachmentState blendStates[1]{};
		blendStates[0].blendEnable = VK_FALSE;
		blendStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

		VkPipelineColorBlendStateCreateInfo blendInfo{};
		blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blendInfo.logicOpEnable = VK_FALSE;
		blendInfo.attachmentCount = 1;
		blendInfo.pAttachments = blendStates;

		VkStencilOpState stencilState{};
		stencilState.compareOp = VK_COMPARE_OP_ALWAYS;
		stencilState.passOp = VK_STENCIL_OP_INCREMENT_AND_CLAMP;
		stencilState.failOp = VK_STENCIL_OP_KEEP;
		stencilState.depthFailOp = VK_STENCIL_OP_KEEP;
		stencilState.writeMask = 0xff;
		stencilState.compareMask = 0xff;
		stencilState.reference = 0x01;

		VkPipelineDepthStencilStateCreateInfo depthInfo{};
		depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthInfo.depthTestEnable = VK_FALSE;
		depthInfo.depthWriteEnable = VK_TRUE;
		depthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthInfo.minDepthBounds = 0.0f;
		depthInfo.maxDepthBounds = 1.0f;
		depthInfo.stencilTestEnable = VK_TRUE;
		depthInfo.front = stencilState;

		VkDynamicState dynamicState{
			VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE
		};

		VkPipelineDynamicStateCreateInfo dynamicInfo{};
		dynamicInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicInfo.dynamicStateCount = 1;
		dynamicInfo.pDynamicStates = &dynamicState;

		VkGraphicsPipelineCreateInfo pipeInfo{};
		pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipeInfo.stageCount = 2;
		pipeInfo.pStages = stages;
		pipeInfo.pVertexInputState = &inputInfo;
		pipeInfo.pInputAssemblyState = &assemblyInfo;
		pipeInfo.pTessellationState = nullptr;
		pipeInfo.pViewportState = &viewportInfo;
		pipeInfo.pRasterizationState = &rasterInfo;
		pipeInfo.pMultisampleState = &samplingInfo;
		pipeInfo.pDepthStencilState = &depthInfo;
		pipeInfo.pColorBlendState = &blendInfo;
		pipeInfo.pDynamicState = &dynamicInfo;
		pipeInfo.layout = aPipelineWriteLayout;
		pipeInfo.renderPass = aRenderPass;
		pipeInfo.subpass = 0;

		VkPipeline writepipe = VK_NULL_HANDLE;
		if (const auto res = vkCreateGraphicsPipelines(aWindow.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &writepipe); VK_SUCCESS != res) {
			throw lut::Error("Unable to create graphics pipeline\n vkCreateGraphicsPipeline() returned %s", lut::to_string(res).c_str());
		}

		lut::ShaderModule vertRead = lut::load_shader_module(aWindow, cfg::kOverVisReadVertShaderPath);
		lut::ShaderModule fragRead = lut::load_shader_module(aWindow, cfg::kOverVisReadFragShaderPath);

		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vertRead.handle;
		stages[0].pName = "main";

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = fragRead.handle;
		stages[1].pName = "main";

		VkPipelineVertexInputStateCreateInfo emptyVertexState{};
		emptyVertexState.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

		rasterInfo.cullMode = VK_CULL_MODE_NONE;
		depthInfo.depthWriteEnable = VK_FALSE;

		pipeInfo.pVertexInputState = &emptyVertexState;
		pipeInfo.layout = aPipelineReadLayout;
		pipeInfo.subpass = 1;		

		VkPipeline readpipe = VK_NULL_HANDLE;
		if (const auto res = vkCreateGraphicsPipelines(aWindow.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &readpipe); VK_SUCCESS != res) {
			throw lut::Error("Unable to create graphics pipeline\n vkCreateGraphicsPipeline() returned %s", lut::to_string(res).c_str());
		}

		return { std::move(lut::Pipeline(aWindow.device, writepipe)), std::move(lut::Pipeline(aWindow.device, readpipe)) };
	}

	std::tuple<lut::Pipeline, lut::Pipeline> create_deferred_shading_pipeline(const lut::VulkanWindow& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineGBufLayout, VkPipelineLayout aPipelineShadingLayout) {
		lut::ShaderModule vert = lut::load_shader_module(aWindow, cfg::kWriteGBufVertShaderPath);
		lut::ShaderModule frag = lut::load_shader_module(aWindow, cfg::kWriteGBufFragShaderPath);

		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vert.handle;
		stages[0].pName = "main";

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = frag.handle;
		stages[1].pName = "main";

		// Even though we pass the TBN frame, normals are kept in to allow me to compare and get the
		// screenshots to compare against the TBN normal mapping for the report
		VkVertexInputBindingDescription vertexInputs[4]{};
		vertexInputs[0].binding = 0;
		vertexInputs[0].stride = sizeof(float) * 3;
		vertexInputs[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		vertexInputs[1].binding = 1;
		vertexInputs[1].stride = sizeof(float) * 2;
		vertexInputs[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		vertexInputs[2].binding = 2;
		vertexInputs[2].stride = sizeof(float) * 3;
		vertexInputs[2].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		vertexInputs[3].binding = 3;
		vertexInputs[3].stride = sizeof(std::uint32_t) * 1;
		vertexInputs[3].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		VkVertexInputAttributeDescription vertexAttributes[4]{};
		vertexAttributes[0].binding = 0;
		vertexAttributes[0].location = 0;
		vertexAttributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[0].offset = 0;
		vertexAttributes[1].binding = 1;
		vertexAttributes[1].location = 1;
		vertexAttributes[1].format = VK_FORMAT_R32G32_SFLOAT;
		vertexAttributes[1].offset = 0;
		vertexAttributes[2].binding = 2;
		vertexAttributes[2].location = 2;
		vertexAttributes[2].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[2].offset = 0;
		vertexAttributes[3].binding = 3;
		vertexAttributes[3].location = 3;
		vertexAttributes[3].format = VK_FORMAT_A2R10G10B10_UNORM_PACK32;
		vertexAttributes[3].offset = 0;

		VkPipelineVertexInputStateCreateInfo inputInfo{};
		inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		inputInfo.vertexBindingDescriptionCount = 4;
		inputInfo.pVertexBindingDescriptions = vertexInputs;
		inputInfo.vertexAttributeDescriptionCount = 4;
		inputInfo.pVertexAttributeDescriptions = vertexAttributes;

		VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
		assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyInfo.primitiveRestartEnable = VK_FALSE;

		VkViewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = float(aWindow.swapchainExtent.width);
		viewport.height = float(aWindow.swapchainExtent.height);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		VkRect2D scissor{};
		scissor.offset = VkOffset2D{ 0, 0 };
		scissor.extent = aWindow.swapchainExtent;

		VkPipelineViewportStateCreateInfo viewportInfo{};
		viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewport;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = &scissor;

		VkPipelineRasterizationStateCreateInfo rasterInfo{};
		rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterInfo.depthClampEnable = VK_FALSE;
		rasterInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterInfo.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterInfo.depthBiasEnable = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterInfo.lineWidth = 1.0f;

		VkPipelineMultisampleStateCreateInfo samplingInfo{};
		samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		VkPipelineColorBlendAttachmentState blendStates[2]{};
		blendStates[0].blendEnable = VK_FALSE;
		blendStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		blendStates[1].blendEnable = VK_FALSE;
		blendStates[1].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

		VkPipelineColorBlendStateCreateInfo blendInfo{};
		blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blendInfo.logicOpEnable = VK_FALSE;
		blendInfo.attachmentCount = 2;
		blendInfo.pAttachments = blendStates;

		VkPipelineDepthStencilStateCreateInfo depthInfo{};
		depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthInfo.depthTestEnable = VK_TRUE;
		depthInfo.depthWriteEnable = VK_TRUE;
		depthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthInfo.minDepthBounds = 0.0f;
		depthInfo.maxDepthBounds = 1.0f;

		VkDynamicState dynamicState{
			VK_DYNAMIC_STATE_CULL_MODE
		};

		VkPipelineDynamicStateCreateInfo dynamicInfo{};
		dynamicInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicInfo.dynamicStateCount = 1;
		dynamicInfo.pDynamicStates = &dynamicState;

		VkGraphicsPipelineCreateInfo pipeInfo{};
		pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipeInfo.stageCount = 2;
		pipeInfo.pStages = stages;
		pipeInfo.pVertexInputState = &inputInfo;
		pipeInfo.pInputAssemblyState = &assemblyInfo;
		pipeInfo.pTessellationState = nullptr;
		pipeInfo.pViewportState = &viewportInfo;
		pipeInfo.pRasterizationState = &rasterInfo;
		pipeInfo.pMultisampleState = &samplingInfo;
		pipeInfo.pDepthStencilState = &depthInfo;
		pipeInfo.pColorBlendState = &blendInfo;
		pipeInfo.pDynamicState = &dynamicInfo;
		pipeInfo.layout = aPipelineGBufLayout;
		pipeInfo.renderPass = aRenderPass;
		pipeInfo.subpass = 0;

		VkPipeline writepipe = VK_NULL_HANDLE;
		if (const auto res = vkCreateGraphicsPipelines(aWindow.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &writepipe); VK_SUCCESS != res) {
			throw lut::Error("Unable to create graphics pipeline\n vkCreateGraphicsPipeline() returned %s", lut::to_string(res).c_str());
		}

		lut::ShaderModule vertRead = lut::load_shader_module(aWindow, cfg::kDefShadingVertShaderPath);
		lut::ShaderModule fragRead = lut::load_shader_module(aWindow, cfg::kDefShadingFragShaderPath);

		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vertRead.handle;
		stages[0].pName = "main";

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = fragRead.handle;
		stages[1].pName = "main";

		VkPipelineVertexInputStateCreateInfo emptyVertexState{};
		emptyVertexState.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

		VkPipelineColorBlendAttachmentState blendStates2[1]{};
		blendStates2[0].blendEnable = VK_FALSE;
		blendStates2[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

		VkPipelineColorBlendStateCreateInfo blendInfo2{};
		blendInfo2.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blendInfo2.logicOpEnable = VK_FALSE;
		blendInfo2.attachmentCount = 1;
		blendInfo2.pAttachments = blendStates2;

		rasterInfo.cullMode = VK_CULL_MODE_NONE;
		depthInfo.depthWriteEnable = VK_FALSE;

		pipeInfo.pVertexInputState = &emptyVertexState;
		pipeInfo.pColorBlendState = &blendInfo2;
		pipeInfo.layout = aPipelineShadingLayout;
		pipeInfo.subpass = 1;

		VkPipeline shadingpipe = VK_NULL_HANDLE;
		if (const auto res = vkCreateGraphicsPipelines(aWindow.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &shadingpipe); VK_SUCCESS != res) {
			throw lut::Error("Unable to create graphics pipeline\n vkCreateGraphicsPipeline() returned %s", lut::to_string(res).c_str());
		}

		return { std::move(lut::Pipeline(aWindow.device, writepipe)), std::move(lut::Pipeline(aWindow.device, shadingpipe)) };
	}

	lut::Pipeline create_shadow_pipeline(const lut::VulkanWindow& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout) {
		lut::ShaderModule vert = lut::load_shader_module(aWindow, cfg::kShadowOffscreenVertShaderPath);
		lut::ShaderModule frag = lut::load_shader_module(aWindow, cfg::kShadowOffscreenFragShaderPath);

		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vert.handle;
		stages[0].pName = "main";

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = frag.handle;
		stages[1].pName = "main";

		VkVertexInputBindingDescription vertexInputs[1]{};
		vertexInputs[0].binding = 0;
		vertexInputs[0].stride = sizeof(float) * 3;
		vertexInputs[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		VkVertexInputAttributeDescription vertexAttributes[1]{};
		vertexAttributes[0].binding = 0;
		vertexAttributes[0].location = 0;
		vertexAttributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[0].offset = 0;

		VkPipelineVertexInputStateCreateInfo inputInfo{};
		inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		inputInfo.vertexBindingDescriptionCount = 1;
		inputInfo.pVertexBindingDescriptions = vertexInputs;
		inputInfo.vertexAttributeDescriptionCount = 1;
		inputInfo.pVertexAttributeDescriptions = vertexAttributes;

		VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
		assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyInfo.primitiveRestartEnable = VK_FALSE;

		VkViewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = float(2048);
		viewport.height = float(2048);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		VkRect2D scissor{};
		scissor.offset = VkOffset2D{ 0, 0 };
		scissor.extent = VkExtent2D{ 2048, 2048 };

		VkPipelineViewportStateCreateInfo viewportInfo{};
		viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewport;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = &scissor;

		VkPipelineRasterizationStateCreateInfo rasterInfo{};
		rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterInfo.depthClampEnable = VK_FALSE;
		rasterInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterInfo.cullMode = VK_CULL_MODE_NONE;
		rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterInfo.depthBiasEnable = VK_TRUE;
		rasterInfo.depthBiasConstantFactor = 7.0f;
		rasterInfo.depthBiasSlopeFactor = 10.0f;
		rasterInfo.lineWidth = 1.0f;

		VkPipelineMultisampleStateCreateInfo samplingInfo{};
		samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		VkPipelineColorBlendAttachmentState blendStates[1]{};
		blendStates[0].blendEnable = VK_FALSE;
		blendStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

		VkPipelineColorBlendStateCreateInfo blendInfo{};
		blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blendInfo.logicOpEnable = VK_FALSE;
		blendInfo.attachmentCount = 0;
		blendInfo.pAttachments = blendStates;

		VkPipelineDepthStencilStateCreateInfo depthInfo{};
		depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthInfo.depthTestEnable = VK_TRUE;
		depthInfo.depthWriteEnable = VK_TRUE;
		depthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthInfo.minDepthBounds = 0.0f;
		depthInfo.maxDepthBounds = 1.0f;

		VkGraphicsPipelineCreateInfo pipeInfo{};
		pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipeInfo.stageCount = 2;
		pipeInfo.pStages = stages;
		pipeInfo.pVertexInputState = &inputInfo;
		pipeInfo.pInputAssemblyState = &assemblyInfo;
		pipeInfo.pTessellationState = nullptr;
		pipeInfo.pViewportState = &viewportInfo;
		pipeInfo.pRasterizationState = &rasterInfo;
		pipeInfo.pMultisampleState = &samplingInfo;
		pipeInfo.pDepthStencilState = &depthInfo;
		pipeInfo.pColorBlendState = &blendInfo;
		pipeInfo.pDynamicState = nullptr;
		pipeInfo.layout = aPipelineLayout;
		pipeInfo.renderPass = aRenderPass;
		pipeInfo.subpass = 0;

		VkPipeline pipe = VK_NULL_HANDLE;
		if (const auto res = vkCreateGraphicsPipelines(aWindow.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipe); VK_SUCCESS != res) {
			throw lut::Error("Unable to create graphics pipeline\n vkCreateGraphicsPipeline() returned %s", lut::to_string(res).c_str());
		}

		return lut::Pipeline(aWindow.device, pipe);
	}

	std::tuple<lut::Image, lut::ImageView> create_depth_buffer(const lut::VulkanWindow& aWindow, const lut::Allocator& aAllocator, VkImageAspectFlagBits aImageAspectFlagBits) {
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D,
		imageInfo.format = cfg::kDepthFormat;
		imageInfo.extent.width = aWindow.swapchainExtent.width;
		imageInfo.extent.height = aWindow.swapchainExtent.height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

		VkImage image = VK_NULL_HANDLE;
		VmaAllocation allocation = VK_NULL_HANDLE;

		if (const auto res = vmaCreateImage(aAllocator.allocator, &imageInfo, &allocInfo, &image, &allocation, nullptr); VK_SUCCESS != res)
			throw lut::Error("Unable to allocate depth buffer image.\n vmaCreateImage() returned %s", lut::to_string(res).c_str());

		lut::Image depthImage(aAllocator.allocator, image, allocation);

		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = depthImage.image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = cfg::kDepthFormat;
		viewInfo.components = VkComponentMapping{};
		viewInfo.subresourceRange = VkImageSubresourceRange{
			(unsigned int)aImageAspectFlagBits,
			0,
			1,
			0,
			1
		};

		VkImageView view = VK_NULL_HANDLE;
		if (const auto res = vkCreateImageView(aWindow.device, &viewInfo, nullptr, &view); VK_SUCCESS != res)
			throw lut::Error("Unable to create image view.\n vkCreateImageView() returned %s", lut::to_string(res).c_str());

		return {std::move(depthImage), lut::ImageView(aWindow.device, view)};
	}

	std::tuple<lut::Image, lut::ImageView> create_colour_buffer(const lut::VulkanWindow& aWindow, const lut::Allocator& aAllocator) {
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D,
		imageInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
		imageInfo.extent.width = aWindow.swapchainExtent.width;
		imageInfo.extent.height = aWindow.swapchainExtent.height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

		VkImage image = VK_NULL_HANDLE;
		VmaAllocation allocation = VK_NULL_HANDLE;

		if (const auto res = vmaCreateImage(aAllocator.allocator, &imageInfo, &allocInfo, &image, &allocation, nullptr); VK_SUCCESS != res)
			throw lut::Error("Unable to allocate depth buffer image.\n vmaCreateImage() returned %s", lut::to_string(res).c_str());

		lut::Image colourImage(aAllocator.allocator, image, allocation);

		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = colourImage.image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
		viewInfo.components = VkComponentMapping{};
		viewInfo.subresourceRange = VkImageSubresourceRange{
			VK_IMAGE_ASPECT_COLOR_BIT,
			0,
			1,
			0,
			1
		};

		VkImageView view = VK_NULL_HANDLE;
		if (const auto res = vkCreateImageView(aWindow.device, &viewInfo, nullptr, &view); VK_SUCCESS != res)
			throw lut::Error("Unable to create image view.\n vkCreateImageView() returned %s", lut::to_string(res).c_str());

		lut::ImageView imgView = lut::ImageView(aWindow.device, view);

		return {std::move(colourImage), std::move(imgView)};
	}

	std::tuple<lut::Image, lut::ImageView> create_just_stencil_buffer(const lut::VulkanWindow& aWindow, const lut::Allocator& aAllocator) {
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D,
		imageInfo.format = VK_FORMAT_S8_UINT;
		imageInfo.extent.width = aWindow.swapchainExtent.width;
		imageInfo.extent.height = aWindow.swapchainExtent.height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

		VkImage image = VK_NULL_HANDLE;
		VmaAllocation allocation = VK_NULL_HANDLE;

		if (const auto res = vmaCreateImage(aAllocator.allocator, &imageInfo, &allocInfo, &image, &allocation, nullptr); VK_SUCCESS != res)
			throw lut::Error("Unable to allocate depth buffer image.\n vmaCreateImage() returned %s", lut::to_string(res).c_str());

		lut::Image stencilImage(aAllocator.allocator, image, allocation);

		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = stencilImage.image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = VK_FORMAT_S8_UINT;
		viewInfo.components = VkComponentMapping{};
		viewInfo.subresourceRange = VkImageSubresourceRange{
			VK_IMAGE_ASPECT_STENCIL_BIT,
			0,
			1,
			0,
			1
		};

		VkImageView view = VK_NULL_HANDLE;
		if (const auto res = vkCreateImageView(aWindow.device, &viewInfo, nullptr, &view); VK_SUCCESS != res)
			throw lut::Error("Unable to create image view.\n vkCreateImageView() returned %s", lut::to_string(res).c_str());

		return { std::move(stencilImage), lut::ImageView(aWindow.device, view) };
	}

	std::tuple<lut::Image, lut::ImageView> create_normals_buffer(const lut::VulkanWindow& aWindow, const lut::Allocator& aAllocator) {
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D,
		imageInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
		imageInfo.extent.width = aWindow.swapchainExtent.width;
		imageInfo.extent.height = aWindow.swapchainExtent.height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

		VkImage image = VK_NULL_HANDLE;
		VmaAllocation allocation = VK_NULL_HANDLE;

		if (const auto res = vmaCreateImage(aAllocator.allocator, &imageInfo, &allocInfo, &image, &allocation, nullptr); VK_SUCCESS != res)
			throw lut::Error("Unable to allocate depth buffer image.\n vmaCreateImage() returned %s", lut::to_string(res).c_str());

		lut::Image normalsImage(aAllocator.allocator, image, allocation);

		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = normalsImage.image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
		viewInfo.components = VkComponentMapping{};
		viewInfo.subresourceRange = VkImageSubresourceRange{
			VK_IMAGE_ASPECT_COLOR_BIT,
			0,
			1,
			0,
			1
		};

		VkImageView view = VK_NULL_HANDLE;
		if (const auto res = vkCreateImageView(aWindow.device, &viewInfo, nullptr, &view); VK_SUCCESS != res)
			throw lut::Error("Unable to create image view.\n vkCreateImageView() returned %s", lut::to_string(res).c_str());

		return { std::move(normalsImage), lut::ImageView(aWindow.device, view) };
	}

	std::tuple<lut::Image, lut::ImageView> create_albedo_buffer(const lut::VulkanWindow& aWindow, const lut::Allocator& aAllocator) {
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D,
		imageInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
		imageInfo.extent.width = aWindow.swapchainExtent.width;
		imageInfo.extent.height = aWindow.swapchainExtent.height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

		VkImage image = VK_NULL_HANDLE;
		VmaAllocation allocation = VK_NULL_HANDLE;

		if (const auto res = vmaCreateImage(aAllocator.allocator, &imageInfo, &allocInfo, &image, &allocation, nullptr); VK_SUCCESS != res)
			throw lut::Error("Unable to allocate depth buffer image.\n vmaCreateImage() returned %s", lut::to_string(res).c_str());

		lut::Image albedoImage(aAllocator.allocator, image, allocation);

		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = albedoImage.image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
		viewInfo.components = VkComponentMapping{};
		viewInfo.subresourceRange = VkImageSubresourceRange{
			VK_IMAGE_ASPECT_COLOR_BIT,
			0,
			1,
			0,
			1
		};

		VkImageView view = VK_NULL_HANDLE;
		if (const auto res = vkCreateImageView(aWindow.device, &viewInfo, nullptr, &view); VK_SUCCESS != res)
			throw lut::Error("Unable to create image view.\n vkCreateImageView() returned %s", lut::to_string(res).c_str());

		return { std::move(albedoImage), lut::ImageView(aWindow.device, view) };
	}

	std::tuple<lut::Image, lut::ImageView> create_shadow_depth_buffer(const lut::VulkanWindow& aWindow, const lut::Allocator& aAllocator) {
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D,
		imageInfo.format = cfg::kDepthFormat;
		imageInfo.extent.width = 2048;
		imageInfo.extent.height = 2048;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

		VkImage image = VK_NULL_HANDLE;
		VmaAllocation allocation = VK_NULL_HANDLE;

		if (const auto res = vmaCreateImage(aAllocator.allocator, &imageInfo, &allocInfo, &image, &allocation, nullptr); VK_SUCCESS != res)
			throw lut::Error("Unable to allocate depth buffer image.\n vmaCreateImage() returned %s", lut::to_string(res).c_str());

		lut::Image depthImage(aAllocator.allocator, image, allocation);

		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = depthImage.image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = cfg::kDepthFormat;
		viewInfo.components = VkComponentMapping{};
		viewInfo.subresourceRange = VkImageSubresourceRange{
			VK_IMAGE_ASPECT_DEPTH_BIT,
			0,
			1,
			0,
			1
		};

		VkImageView view = VK_NULL_HANDLE;
		if (const auto res = vkCreateImageView(aWindow.device, &viewInfo, nullptr, &view); VK_SUCCESS != res)
			throw lut::Error("Unable to create image view.\n vkCreateImageView() returned %s", lut::to_string(res).c_str());

		return { std::move(depthImage), lut::ImageView(aWindow.device, view) };
	}

	lut::Framebuffer create_offscreen_framebuffer(const lut::VulkanWindow& aWindow, VkRenderPass aRenderPass, VkImageView aColourView, VkImageView aDepthView) {
		VkImageView attachments[2] = {
			aColourView,
			aDepthView
		};

		VkFramebufferCreateInfo fbInfo{};
		fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		fbInfo.flags = 0;
		fbInfo.renderPass = aRenderPass;
		fbInfo.attachmentCount = 2;
		fbInfo.pAttachments = attachments;
		fbInfo.width = aWindow.swapchainExtent.width;
		fbInfo.height = aWindow.swapchainExtent.height;
		fbInfo.layers = 1;

		VkFramebuffer fb = VK_NULL_HANDLE;
		if (const auto res = vkCreateFramebuffer(aWindow.device, &fbInfo, nullptr, &fb); VK_SUCCESS != res)
			throw lut::Error("Unable to create framebuffer\n vkCreateFramebuffer() returned %s", lut::to_string(res).c_str());

		return lut::Framebuffer(aWindow.device, fb);
	}

	void create_regular_swapchain_framebuffers(const lut::VulkanWindow& aWindow, VkRenderPass aRenderPass, std::vector<lut::Framebuffer>& aFramebuffers, VkImageView aDepthView) {
		assert( aFramebuffers.empty() );

		for (std::size_t i = 0; i < aWindow.swapViews.size(); ++i) {
			VkImageView attachments[2] = {
				aWindow.swapViews[i],
				aDepthView
			};

			VkFramebufferCreateInfo fbInfo{};
			fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			fbInfo.flags = 0;
			fbInfo.renderPass = aRenderPass;
			fbInfo.attachmentCount = 2;
			fbInfo.pAttachments = attachments;
			fbInfo.width = aWindow.swapchainExtent.width;
			fbInfo.height = aWindow.swapchainExtent.height;
			fbInfo.layers = 1;

			VkFramebuffer fb = VK_NULL_HANDLE;
			if (const auto res = vkCreateFramebuffer(aWindow.device, &fbInfo, nullptr, &fb); VK_SUCCESS != res)
				throw lut::Error("Unable to create framebuffer for swap chain image %zu\n vkCreateFramebuffer() returned %s", i, lut::to_string(res).c_str());

			aFramebuffers.emplace_back(lut::Framebuffer(aWindow.device, fb));
		}

		assert( aWindow.swapViews.size() == aFramebuffers.size() );
	}

	void create_fullscreen_swapchain_framebuffers(const lut::VulkanWindow& aWindow, VkRenderPass aRenderPass, std::vector<lut::Framebuffer>& aFramebuffers) {
		assert( aFramebuffers.empty() );

		for (std::size_t i = 0; i < aWindow.swapViews.size(); ++i) {
			VkImageView attachments[1] = {
				aWindow.swapViews[i]
			};

			VkFramebufferCreateInfo fbInfo{};
			fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			fbInfo.flags = 0;
			fbInfo.renderPass = aRenderPass;
			fbInfo.attachmentCount = 1;
			fbInfo.pAttachments = attachments;
			fbInfo.width = aWindow.swapchainExtent.width;
			fbInfo.height = aWindow.swapchainExtent.height;
			fbInfo.layers = 1;

			VkFramebuffer fb = VK_NULL_HANDLE;
			if (const auto res = vkCreateFramebuffer(aWindow.device, &fbInfo, nullptr, &fb); VK_SUCCESS != res)
				throw lut::Error("Unable to create framebuffer for swap chain image %zu\n vkCreateFramebuffer() returned %s", i, lut::to_string(res).c_str());

			aFramebuffers.emplace_back(lut::Framebuffer(aWindow.device, fb));
		}

		assert( aWindow.swapViews.size() == aFramebuffers.size() );
	}

	void create_over_visualisation_framebuffers(const lut::VulkanWindow& aWindow, VkRenderPass aRenderPass, std::vector<lut::Framebuffer>& aFramebuffers, VkImageView aColorView, VkImageView aDepthStencilView) {
		assert(aFramebuffers.empty());

		for (std::size_t i = 0; i < aWindow.swapViews.size(); ++i) {
			VkImageView attachments[3] = {
				aWindow.swapViews[i],
				aColorView,
				aDepthStencilView
			};

			VkFramebufferCreateInfo fbInfo{};
			fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			fbInfo.flags = 0;
			fbInfo.renderPass = aRenderPass;
			fbInfo.attachmentCount = 3;
			fbInfo.pAttachments = attachments;
			fbInfo.width = aWindow.swapchainExtent.width;
			fbInfo.height = aWindow.swapchainExtent.height;
			fbInfo.layers = 1;

			VkFramebuffer fb = VK_NULL_HANDLE;
			if (const auto res = vkCreateFramebuffer(aWindow.device, &fbInfo, nullptr, &fb); VK_SUCCESS != res)
				throw lut::Error("Unable to create framebuffer for swap chain image %zu\n vkCreateFramebuffer() returned %s", i, lut::to_string(res).c_str());

			aFramebuffers.emplace_back(lut::Framebuffer(aWindow.device, fb));
		}

		assert(aWindow.swapViews.size() == aFramebuffers.size());
	}

	void create_deferred_shading_framebuffers(const lut::VulkanWindow& aWindow, VkRenderPass aRenderPass, std::vector<lut::Framebuffer>& aFramebuffers, VkImageView aDepthView, VkImageView aNormalsView, VkImageView aAlbedoView) {
		assert(aFramebuffers.empty());

		for (std::size_t i = 0; i < aWindow.swapViews.size(); ++i) {
			VkImageView attachments[4] = {
				aWindow.swapViews[i],
				aNormalsView,
				aAlbedoView,
				aDepthView
			};

			VkFramebufferCreateInfo fbInfo{};
			fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			fbInfo.flags = 0;
			fbInfo.renderPass = aRenderPass;
			fbInfo.attachmentCount = 4;
			fbInfo.pAttachments = attachments;
			fbInfo.width = aWindow.swapchainExtent.width;
			fbInfo.height = aWindow.swapchainExtent.height;
			fbInfo.layers = 1;

			VkFramebuffer fb = VK_NULL_HANDLE;
			if (const auto res = vkCreateFramebuffer(aWindow.device, &fbInfo, nullptr, &fb); VK_SUCCESS != res)
				throw lut::Error("Unable to create framebuffer for swap chain image %zu\n vkCreateFramebuffer() returned %s", i, lut::to_string(res).c_str());

			aFramebuffers.emplace_back(lut::Framebuffer(aWindow.device, fb));
		}

		assert(aWindow.swapViews.size() == aFramebuffers.size());
	}

	void create_shadow_offscreen_framebuffers(const lut::VulkanWindow& aWindow, VkRenderPass aRenderPass, std::vector<lut::Framebuffer>& aFramebuffers, VkImageView aDepthView) {
		assert(aFramebuffers.empty());

		for (std::size_t i = 0; i < aWindow.swapViews.size(); ++i) {
			VkImageView attachments = aDepthView;

			VkFramebufferCreateInfo fbInfo{};
			fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			fbInfo.flags = 0;
			fbInfo.renderPass = aRenderPass;
			fbInfo.attachmentCount = 1;
			fbInfo.pAttachments = &attachments;
			fbInfo.width = 2048;
			fbInfo.height = 2048;
			fbInfo.layers = 1;

			VkFramebuffer fb = VK_NULL_HANDLE;
			if (const auto res = vkCreateFramebuffer(aWindow.device, &fbInfo, nullptr, &fb); VK_SUCCESS != res)
				throw lut::Error("Unable to create framebuffer\n vkCreateFramebuffer() returned %s", lut::to_string(res).c_str());

			aFramebuffers.emplace_back(lut::Framebuffer(aWindow.device, fb));
		}

		assert(aWindow.swapViews.size() == aFramebuffers.size());
	}

	lut::ImageView load_mesh_texture(const lut::VulkanWindow& aWindow, VkCommandPool aCmdPool, const lut::Allocator& aAllocator, BakedTextureInfo aBakedTextureInfo) {
		VkFormat format = aBakedTextureInfo.space == ETextureSpace::srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;

		// printf("channels: %d\n", aBakedTextureInfo.channels);
		
		lut::Image image = lut::load_image_texture2d(aBakedTextureInfo.path.c_str(), aWindow, aCmdPool, aAllocator, format, aBakedTextureInfo.channels);
		
		lut::ImageView imageView = lut::create_image_view_texture2d(aWindow, image.image, format);
		images.emplace_back(std::move(image));

		return imageView;
	}

	lut::ImageView get_dummy_texture(const lut::VulkanWindow& aWindow, VkCommandPool aCmdPool, const lut::Allocator& aAllocator) {
		lut::Image image = lut::get_dummy_image(aWindow, aAllocator, aCmdPool, VK_FORMAT_R8G8B8A8_UNORM);

		lut::ImageView imageView = lut::create_image_view_texture2d(aWindow, image.image, VK_FORMAT_R8G8B8A8_UNORM);
		images.push_back(std::move(image));

		return imageView;
	}

	void update_user_state(UserState& aState, float aElapsedTime) {
		auto& cam = aState.camera2world;

		if (aState.inputMap[std::size_t(EInputState::mousing)]) {
			if (aState.wasMousing) {
				const auto sens = cfg::kCameraMouseSensitivity;
				const auto dx = sens * (aState.mouseX - aState.previousX);
				const auto dy = sens * (aState.mouseY - aState.previousY);
			
				cam = cam * glm::rotate(-dy, glm::vec3(1.0f, 0.0f, 0.0f));
				cam = cam * glm::rotate(-dx, glm::vec3(0.0f, 1.0f, 0.0f));
			}

			aState.previousX = aState.mouseX;
			aState.previousY = aState.mouseY;
			aState.wasMousing = true;
		} else {
			aState.wasMousing = false;
		}

		const auto move = aElapsedTime * cfg::kCameraBaseSpeed *
			(aState.inputMap[std::size_t(EInputState::fast)] ? cfg::kCameraFastMult : 1.0f) *
			(aState.inputMap[std::size_t(EInputState::slow)] ? cfg::kCameraSlowMult : 1.0f);

		if( aState.inputMap[std::size_t(EInputState::forward)] )
			cam = cam * glm::translate( glm::vec3( 0.f, 0.f, -move ) );
		if( aState.inputMap[std::size_t(EInputState::backward)] )
		 	cam = cam * glm::translate( glm::vec3( 0.f, 0.f, +move ) );
		if( aState.inputMap[std::size_t(EInputState::strafeLeft)] )
		 	cam = cam * glm::translate( glm::vec3( -move, 0.f, 0.f ) );
		if( aState.inputMap[std::size_t(EInputState::strafeRight)] )
		 	cam = cam * glm::translate( glm::vec3( +move, 0.f, 0.f ) );
		if( aState.inputMap[std::size_t(EInputState::levitate)] )
		 	cam = cam * glm::translate( glm::vec3( 0.f, +move, 0.f ) );
		if( aState.inputMap[std::size_t(EInputState::sink)] )
		 	cam = cam * glm::translate( glm::vec3( 0.f, -move, 0.f ) );
	}

	void update_scene_uniforms( glsl::SceneUniform& aSceneUniforms, std::uint32_t aFramebufferWidth, std::uint32_t aFramebufferHeight, UserState const& aState ) {
		const float aspect = aFramebufferWidth / float(aFramebufferHeight);

		aSceneUniforms.projection = glm::perspectiveRH_ZO(lut::Radians(cfg::kCameraFov).value(), aspect, cfg::kCameraNear, cfg::kCameraFar);
		aSceneUniforms.projection[1][1] *= -1.0f;
		aSceneUniforms.camera = glm::inverse(aState.camera2world);
		//aSceneUniforms.projection = glm::orthoRH_ZO(-10.0f, 10.0f, 10.0f, -10.0f, 0.1f, 1000.0f);
		//aSceneUniforms.projection = glm::perspectiveRH_ZO(lut::Radians(90.0f).value(), 1.0f, cfg::kCameraNear, cfg::kCameraFar);
		//aSceneUniforms.projection[1][1] *= -1.0f;
		//aSceneUniforms.camera = glm::lookAt(glm::vec3(-0.2972f, 7.3100f, -11.9532f), glm::vec3(0.0f, 0.0f, -48.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		aSceneUniforms.projCam = aSceneUniforms.projection * aSceneUniforms.camera;
		aSceneUniforms.camPos = glm::vec4(aState.camera2world[3][0], aState.camera2world[3][1], aState.camera2world[3][2], 1.0f);
	}

	void update_debug_uniforms(glsl::DebugUniform& aDebugUniform, const UserState& aState) {
		aDebugUniform.debug = aState.debugVisualisation;
	}

	void update_depth_mvp_uniforms(glsl::DepthMVP& aDepthUniform, std::uint32_t aFramebufferWidth, std::uint32_t aFramebufferHeight, glm::vec4 lightPos) {
		const float aspect = aFramebufferWidth / float(aFramebufferHeight);

		glm::mat4 depthProjection = glm::perspectiveRH_ZO(lut::Radians(90.0f).value(), 1.0f, cfg::kCameraNear, cfg::kCameraFar);
		depthProjection[1][1] *= -1.0f;
		glm::mat4 depthView = glm::lookAt(glm::vec3(lightPos), glm::vec3(0.0f, 0.0f, -48.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		
		aDepthUniform.depthMVP = depthProjection * depthView;
	}

	void record_commands(
		VkCommandBuffer aCmdBuff,
		RenderPasses aRenderPasses,
		Framebuffers aFramebuffers,
		Pipelines aPipelines,
		const VkExtent2D& aExtent,
		std::vector<MeshData>& aMeshData,
		UBOs aUBOs,
		Uniforms aUniforms,
		PipelineLayouts aPipelineLayouts,
		DescriptorSets aDescriptorSets,
		const UserState& aState
	) {
		VkCommandBufferBeginInfo begInfo{};
		begInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		begInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		begInfo.pInheritanceInfo = nullptr;

		if (const auto res = vkBeginCommandBuffer(aCmdBuff, &begInfo); VK_SUCCESS != res) {
			throw lut::Error("Unable to begin recording command buffer\n vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str());
		}

		// Scene UBO
		lut::buffer_barrier(
			aCmdBuff,
			aUBOs.sceneUBO,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT
		);

		vkCmdUpdateBuffer(aCmdBuff, aUBOs.sceneUBO, 0, sizeof(glsl::SceneUniform), &aUniforms.sceneUniforms);

		lut::buffer_barrier(
			aCmdBuff,
			aUBOs.sceneUBO,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
		);

		// Light UBO
		lut::buffer_barrier(
			aCmdBuff,
			aUBOs.lightUBO,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT
		);
		
		vkCmdUpdateBuffer(aCmdBuff, aUBOs.lightUBO, 0, sizeof(glsl::LightUniform), &aUniforms.lightUniforms);

		lut::buffer_barrier(
			aCmdBuff,
			aUBOs.lightUBO,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
		);

		// Debug UBO
		lut::buffer_barrier(
			aCmdBuff,
			aUBOs.debugUBO,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT
		);

		vkCmdUpdateBuffer(aCmdBuff, aUBOs.debugUBO, 0, sizeof(glsl::DebugUniform), &aUniforms.debugUniforms);

		lut::buffer_barrier(
			aCmdBuff,
			aUBOs.debugUBO,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
		);

		// Lights UBO
		lut::buffer_barrier(
			aCmdBuff,
			aUBOs.multipleLightsUBO,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT
		);

		vkCmdUpdateBuffer(aCmdBuff, aUBOs.multipleLightsUBO, 0, sizeof(glsl::MultipleLights), &aUniforms.multipleLightsUniform);

		lut::buffer_barrier(
			aCmdBuff,
			aUBOs.multipleLightsUBO,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
		);

		lut::buffer_barrier(
			aCmdBuff,
			aUBOs.depthMVPUBO,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT
		);

		vkCmdUpdateBuffer(aCmdBuff, aUBOs.depthMVPUBO, 0, sizeof(glsl::DepthMVP), &aUniforms.depthMVPUniform);

		lut::buffer_barrier(
			aCmdBuff,
			aUBOs.depthMVPUBO,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_SHADER_BIT
		);

		// Deferred shading
		if (aState.deferredShading && !aState.mosaicEffect) {
			VkClearValue clearValues[4]{};
			clearValues[0].color.float32[0] = 0.1f;
			clearValues[0].color.float32[1] = 0.1f;
			clearValues[0].color.float32[2] = 0.1f;
			clearValues[0].color.float32[3] = 0.1f;

			clearValues[1].color.float32[0] = 0.1f;
			clearValues[1].color.float32[1] = 0.1f;
			clearValues[1].color.float32[2] = 0.1f;
			clearValues[1].color.float32[3] = 0.1f;

			clearValues[2].color.float32[0] = 0.1f;
			clearValues[2].color.float32[1] = 0.1f;
			clearValues[2].color.float32[2] = 0.1f;
			clearValues[2].color.float32[3] = 0.1f;			
			
			clearValues[3].depthStencil.depth = 1.0f;

			VkRenderPassBeginInfo passInfo{};
			passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			passInfo.renderPass = aRenderPasses.deferredShadingRenderPass;
			passInfo.framebuffer = aFramebuffers.deferredShadingFramebuffer;
			passInfo.renderArea.offset = VkOffset2D{ 0, 0 };
			passInfo.renderArea.extent = aExtent;
			passInfo.clearValueCount = 4;
			passInfo.pClearValues = clearValues;

			vkCmdBeginRenderPass(aCmdBuff, &passInfo, VK_SUBPASS_CONTENTS_INLINE);

			vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aPipelines.gBufWritePipline);

			vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aPipelineLayouts.gBufWritePipelineLayout, 0, 1, &aDescriptorSets.sceneDescriptors, 0, nullptr);

			vkCmdSetCullMode(aCmdBuff, VK_CULL_MODE_BACK_BIT);

			// Draw all non alpha masked meshes
			for (std::size_t i = 0; i < aMeshData.size(); i++) {
				if (aMeshData[i].hasAlphaMask) continue;

				vkCmdBindDescriptorSets(
					aCmdBuff,
					VK_PIPELINE_BIND_POINT_GRAPHICS,
					aPipelineLayouts.gBufWritePipelineLayout,
					1,
					1,
					&aDescriptorSets.materialDescriptors[aMeshData[i].materialId],
					0,
					nullptr
				);

				VkBuffer vbuffers[4] = {
					aMeshData[i].positionBuffer.buffer,
					aMeshData[i].texCoordBuffer.buffer,
					aMeshData[i].normalsBuffer.buffer,
					aMeshData[i].tangentsBuffer.buffer
				};
				VkBuffer ibuffer = aMeshData[i].indicesBuffer.buffer;
				VkDeviceSize voffsets[4]{};
				VkDeviceSize ioffset{};

				vkCmdBindVertexBuffers(aCmdBuff, 0, 4, vbuffers, voffsets);
				vkCmdBindIndexBuffer(aCmdBuff, ibuffer, ioffset, VK_INDEX_TYPE_UINT32);

				vkCmdDrawIndexed(aCmdBuff, aMeshData[i].indicesCount, 1, 0, 0, 0);
			}

			// We dont want to cull back faces for alpha masked meshes since those are the foliage and we want both sides of the mesh
			vkCmdSetCullMode(aCmdBuff, VK_CULL_MODE_NONE);

			for (std::size_t i = 0; i < aMeshData.size(); i++) {
				if (!aMeshData[i].hasAlphaMask) continue;

				vkCmdBindDescriptorSets(
					aCmdBuff,
					VK_PIPELINE_BIND_POINT_GRAPHICS,
					aPipelineLayouts.gBufWritePipelineLayout,
					1,
					1,
					&aDescriptorSets.materialDescriptors[aMeshData[i].materialId],
					0,
					nullptr
				);

				VkBuffer vbuffers[4] = {
					aMeshData[i].positionBuffer.buffer,
					aMeshData[i].texCoordBuffer.buffer,
					aMeshData[i].normalsBuffer.buffer,
					aMeshData[i].tangentsBuffer.buffer
				};
				VkBuffer ibuffer = aMeshData[i].indicesBuffer.buffer;
				VkDeviceSize voffsets[4]{};
				VkDeviceSize ioffset{};

				vkCmdBindVertexBuffers(aCmdBuff, 0, 4, vbuffers, voffsets);
				vkCmdBindIndexBuffer(aCmdBuff, ibuffer, ioffset, VK_INDEX_TYPE_UINT32);

				vkCmdDrawIndexed(aCmdBuff, aMeshData[i].indicesCount, 1, 0, 0, 0);
			}

			vkCmdNextSubpass(aCmdBuff, VK_SUBPASS_CONTENTS_INLINE);

			vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aPipelines.deferredShadingPipeline);
			vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aPipelineLayouts.deferredShadingPipelineLayout, 0, 1, &aDescriptorSets.deferredShadingDescriptor, 0, nullptr);
			vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aPipelineLayouts.deferredShadingPipelineLayout, 1, 1, &aDescriptorSets.sceneDescriptors, 0, nullptr);
			vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aPipelineLayouts.deferredShadingPipelineLayout, 2, 1, &aDescriptorSets.multipleLightsDescriptor, 0, nullptr);

			vkCmdDraw(aCmdBuff, 3, 1, 0, 0);

			vkCmdEndRenderPass(aCmdBuff);
		}
		// Setup regular rendering (no debug, no mosaic)
		else if (!aState.mosaicEffect && aState.debugVisualisation == 1) {
			// Shadow pass
			VkClearValue clearValuesS{};
			clearValuesS.depthStencil.depth = 1.0f;

			VkRenderPassBeginInfo passInfoS{};
			passInfoS.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			passInfoS.renderPass = aRenderPasses.shadowOffscreenRenderPass;
			passInfoS.framebuffer = aFramebuffers.shadowOffscreenFramebuffer;
			passInfoS.renderArea.offset = VkOffset2D{ 0, 0 };
			passInfoS.renderArea.extent = VkExtent2D{ 2048, 2048 };
			passInfoS.clearValueCount = 1;
			passInfoS.pClearValues = &clearValuesS;

			vkCmdBeginRenderPass(aCmdBuff, &passInfoS, VK_SUBPASS_CONTENTS_INLINE);

			vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aPipelines.shadowOffscreenPipeline);
			vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aPipelineLayouts.shadowOffscreenPipelineLayout, 0, 1, &aDescriptorSets.depthMVPDescriptor, 0, nullptr);

			// Draw all non alpha masked meshes
			for (std::size_t i = 0; i < aMeshData.size(); i++) {
				if (aMeshData[i].hasAlphaMask) continue;

				VkBuffer vbuffers[1] = {
					aMeshData[i].positionBuffer.buffer
				};
				VkBuffer ibuffer = aMeshData[i].indicesBuffer.buffer;
				VkDeviceSize voffsets[1]{};
				VkDeviceSize ioffset{};

				vkCmdBindVertexBuffers(aCmdBuff, 0, 1, vbuffers, voffsets);
				vkCmdBindIndexBuffer(aCmdBuff, ibuffer, ioffset, VK_INDEX_TYPE_UINT32);

				vkCmdDrawIndexed(aCmdBuff, aMeshData[i].indicesCount, 1, 0, 0, 0);
			}

			vkCmdEndRenderPass(aCmdBuff);

			// Default rendering
			VkClearValue clearValues[2]{};
			clearValues[0].color.float32[0] = 0.1f;
			clearValues[0].color.float32[1] = 0.1f;
			clearValues[0].color.float32[2] = 0.1f;
			clearValues[0].color.float32[3] = 1.0f;

			clearValues[1].depthStencil.depth = 1.0f;

			VkRenderPassBeginInfo passInfo{};
			passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			passInfo.renderPass = aRenderPasses.regularRenderPass;
			passInfo.framebuffer = aFramebuffers.regularSwapchainFramebuffer;
			passInfo.renderArea.offset = VkOffset2D{0, 0};
			passInfo.renderArea.extent = aExtent;
			passInfo.clearValueCount = 2;
			passInfo.pClearValues = clearValues;

			vkCmdBeginRenderPass(aCmdBuff, &passInfo, VK_SUBPASS_CONTENTS_INLINE);

			vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aPipelines.regularPipeline);

			vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aPipelineLayouts.regularPipelineLayout, 0, 1, &aDescriptorSets.sceneDescriptors, 0, nullptr);
			vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aPipelineLayouts.regularPipelineLayout, 2, 1, &aDescriptorSets.lightDescriptor, 0, nullptr);
			vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aPipelineLayouts.regularPipelineLayout, 3, 1, &aDescriptorSets.depthMVPDescriptor, 0, nullptr);
			vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aPipelineLayouts.regularPipelineLayout, 4, 1, &aDescriptorSets.shadowMapDescriptor, 0, nullptr);

			// Draw all non alpha masked meshes
			for (std::size_t i = 0; i < aMeshData.size(); i++) {
				if (aMeshData[i].hasAlphaMask) continue;

				vkCmdBindDescriptorSets(
					aCmdBuff, 
					VK_PIPELINE_BIND_POINT_GRAPHICS, 
					aPipelineLayouts.regularPipelineLayout, 
					1, 
					1, 
					&aDescriptorSets.materialDescriptors[aMeshData[i].materialId], 
					0,
					nullptr
				);

				VkBuffer vbuffers[4] = {
					aMeshData[i].positionBuffer.buffer,
					aMeshData[i].texCoordBuffer.buffer,
					aMeshData[i].normalsBuffer.buffer,
					aMeshData[i].tangentsBuffer.buffer
				};
				VkBuffer ibuffer = aMeshData[i].indicesBuffer.buffer;
				VkDeviceSize voffsets[4]{};
				VkDeviceSize ioffset{};

				vkCmdBindVertexBuffers(aCmdBuff, 0, 4, vbuffers, voffsets);
				vkCmdBindIndexBuffer(aCmdBuff, ibuffer, ioffset, VK_INDEX_TYPE_UINT32);

				vkCmdDrawIndexed(aCmdBuff, aMeshData[i].indicesCount, 1, 0, 0, 0);
			}

			// Draw all alpha masked meshes
			vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aPipelines.alphaPipeline);

			for (std::size_t i = 0; i < aMeshData.size(); i++) {
				if (!aMeshData[i].hasAlphaMask) continue;

				vkCmdBindDescriptorSets(
					aCmdBuff, 
					VK_PIPELINE_BIND_POINT_GRAPHICS, 
					aPipelineLayouts.regularPipelineLayout, 
					1, 
					1, 
					&aDescriptorSets.materialDescriptors[aMeshData[i].materialId], 
					0,
					nullptr
				);

				VkBuffer vbuffers[4] = {
					aMeshData[i].positionBuffer.buffer,
					aMeshData[i].texCoordBuffer.buffer,
					aMeshData[i].normalsBuffer.buffer,
					aMeshData[i].tangentsBuffer.buffer
				};
				VkBuffer ibuffer = aMeshData[i].indicesBuffer.buffer;
				VkDeviceSize voffsets[4]{};
				VkDeviceSize ioffset{};

				vkCmdBindVertexBuffers(aCmdBuff, 0, 4, vbuffers, voffsets);
				vkCmdBindIndexBuffer(aCmdBuff, ibuffer, ioffset, VK_INDEX_TYPE_UINT32);

				vkCmdDrawIndexed(aCmdBuff, aMeshData[i].indicesCount, 1, 0, 0, 0);
			}

			vkCmdEndRenderPass(aCmdBuff);

		}
		// Overdraw/-shading visualisation
		else if (!aState.mosaicEffect && (aState.debugVisualisation == 5 || aState.debugVisualisation == 6)) {
			VkClearValue clearValues[3]{};
			clearValues[0].color.float32[0] = 0.0f;
			clearValues[0].color.float32[1] = 1.0f;
			clearValues[0].color.float32[2] = 0.0f;
			clearValues[0].color.float32[3] = 1.0f;

			clearValues[1].color.float32[0] = 0.1f;
			clearValues[1].color.float32[1] = 0.1f;
			clearValues[1].color.float32[2] = 0.1f;
			clearValues[1].color.float32[3] = 1.0f;

			clearValues[2].depthStencil.depth = 1.0f;

			VkRenderPassBeginInfo passInfo{};
			passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			passInfo.renderPass = aRenderPasses.overVisualisationsRenderPass;
			passInfo.framebuffer = aFramebuffers.overVisualisationFramebuffer;
			passInfo.renderArea.offset = VkOffset2D{ 0, 0 };
			passInfo.renderArea.extent = aExtent;
			passInfo.clearValueCount = 3;
			passInfo.pClearValues = clearValues;

			vkCmdBeginRenderPass(aCmdBuff, &passInfo, VK_SUBPASS_CONTENTS_INLINE);

			vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aPipelines.overVisWritePipeline);

			// 1st subpass only needs scene descriptors
			vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aPipelineLayouts.overVisWritePipelineLayout, 0, 1, &aDescriptorSets.sceneDescriptors, 0, nullptr);

			for (std::size_t i = 0; i < aMeshData.size(); i++) {
				VkBuffer vbuffers[1] = { aMeshData[i].positionBuffer.buffer };
				VkBuffer ibuffer = aMeshData[i].indicesBuffer.buffer;
				VkDeviceSize voffsets[1]{};
				VkDeviceSize ioffset{};

				vkCmdBindVertexBuffers(aCmdBuff, 0, 1, vbuffers, voffsets);
				vkCmdBindIndexBuffer(aCmdBuff, ibuffer, ioffset, VK_INDEX_TYPE_UINT32);

				if (aState.debugVisualisation == 5) // Overdraw
					vkCmdSetDepthTestEnable(aCmdBuff, VK_FALSE);
				else if (aState.debugVisualisation == 6) // Overshading
					vkCmdSetDepthTestEnable(aCmdBuff, VK_TRUE);

				vkCmdDrawIndexed(aCmdBuff, aMeshData[i].indicesCount, 1, 0, 0, 0);
			}

			vkCmdNextSubpass(aCmdBuff, VK_SUBPASS_CONTENTS_INLINE);

			vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aPipelines.overVisReadPipeline);
			vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aPipelineLayouts.overVisReadPipelineLayout, 0, 1, &aDescriptorSets.overVisualisationDescriptor, 0, nullptr);
			
			vkCmdDraw(aCmdBuff, 3, 1, 0, 0);

			vkCmdEndRenderPass(aCmdBuff);
		}
		// Do debug visualisation rendering
		else if (!aState.mosaicEffect && aState.debugVisualisation != 1) {
			VkClearValue clearValues[2]{};
			clearValues[0].color.float32[0] = 0.1f;
			clearValues[0].color.float32[1] = 0.1f;
			clearValues[0].color.float32[2] = 0.1f;
			clearValues[0].color.float32[3] = 1.0f;

			clearValues[1].depthStencil.depth = 1.0f;

			VkRenderPassBeginInfo passInfo{};
			passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			passInfo.renderPass = aRenderPasses.regularRenderPass;
			passInfo.framebuffer = aFramebuffers.regularSwapchainFramebuffer;
			passInfo.renderArea.offset = VkOffset2D{0, 0};
			passInfo.renderArea.extent = aExtent;
			passInfo.clearValueCount = 2;
			passInfo.pClearValues = clearValues;

			vkCmdBeginRenderPass(aCmdBuff, &passInfo, VK_SUBPASS_CONTENTS_INLINE);

			vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aPipelines.debugPipeline);

			vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aPipelineLayouts.regularPipelineLayout, 0, 1, &aDescriptorSets.sceneDescriptors, 0, nullptr);
			vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aPipelineLayouts.regularPipelineLayout, 2, 1, &aDescriptorSets.debugDescriptor, 0, nullptr);
		
			// Draw all meshes
			for (std::size_t i = 0; i < aMeshData.size(); i++) {
				vkCmdBindDescriptorSets(
					aCmdBuff, 
					VK_PIPELINE_BIND_POINT_GRAPHICS, 
					aPipelineLayouts.regularPipelineLayout, 
					1, 
					1, 
					&aDescriptorSets.materialDescriptors[aMeshData[i].materialId], 
					0,
					nullptr
				);

				VkBuffer vbuffers[4] = {
					aMeshData[i].positionBuffer.buffer,
					aMeshData[i].texCoordBuffer.buffer,
					aMeshData[i].normalsBuffer.buffer,
					aMeshData[i].tangentsBuffer.buffer
				};
				VkBuffer ibuffer = aMeshData[i].indicesBuffer.buffer;
				VkDeviceSize voffsets[4]{};
				VkDeviceSize ioffset{};

				vkCmdBindVertexBuffers(aCmdBuff, 0, 4, vbuffers, voffsets);
				vkCmdBindIndexBuffer(aCmdBuff, ibuffer, ioffset, VK_INDEX_TYPE_UINT32);

				vkCmdDrawIndexed(aCmdBuff, aMeshData[i].indicesCount, 1, 0, 0, 0);
			}

			vkCmdEndRenderPass(aCmdBuff);
		} 
		// Do post processing for mosaic effect (ignores debug visualisation state and forces post processing pipeline)
		else if (aState.mosaicEffect) {
			VkClearValue clearValues[2]{};
			clearValues[0].color.float32[0] = 0.1f;
			clearValues[0].color.float32[1] = 0.1f;
			clearValues[0].color.float32[2] = 0.1f;
			clearValues[0].color.float32[3] = 1.0f;

			clearValues[1].depthStencil.depth = 1.0f;

			VkRenderPassBeginInfo passInfo{};
			passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			passInfo.renderPass = aRenderPasses.offscreenRenderPass;
			passInfo.framebuffer = aFramebuffers.offscreenFramebuffer;
			passInfo.renderArea.offset = VkOffset2D{0, 0};
			passInfo.renderArea.extent = aExtent;
			passInfo.clearValueCount = 2;
			passInfo.pClearValues = clearValues;

			vkCmdBeginRenderPass(aCmdBuff, &passInfo, VK_SUBPASS_CONTENTS_INLINE);

			vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aPipelines.offscreenPipeline);

			vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aPipelineLayouts.regularPipelineLayout, 0, 1, &aDescriptorSets.sceneDescriptors, 0, nullptr);
			vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aPipelineLayouts.regularPipelineLayout, 2, 1, &aDescriptorSets.lightDescriptor, 0, nullptr);
		
			// Draw all non alpha masked meshes
			for (std::size_t i = 0; i < aMeshData.size(); i++) {
				if (aMeshData[i].hasAlphaMask) continue;

				vkCmdBindDescriptorSets(
					aCmdBuff, 
					VK_PIPELINE_BIND_POINT_GRAPHICS, 
					aPipelineLayouts.regularPipelineLayout, 
					1, 
					1, 
					&aDescriptorSets.materialDescriptors[aMeshData[i].materialId], 
					0,
					nullptr
				);

				VkBuffer vbuffers[4] = {
					aMeshData[i].positionBuffer.buffer,
					aMeshData[i].texCoordBuffer.buffer,
					aMeshData[i].normalsBuffer.buffer,
					aMeshData[i].tangentsBuffer.buffer
				};
				VkBuffer ibuffer = aMeshData[i].indicesBuffer.buffer;
				VkDeviceSize voffsets[4]{};
				VkDeviceSize ioffset{};

				vkCmdBindVertexBuffers(aCmdBuff, 0, 4, vbuffers, voffsets);
				vkCmdBindIndexBuffer(aCmdBuff, ibuffer, ioffset, VK_INDEX_TYPE_UINT32);

				vkCmdDrawIndexed(aCmdBuff, aMeshData[i].indicesCount, 1, 0, 0, 0);
			}

			// Draw all alpha masked meshes
			vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aPipelines.alphaOffscreenPipeline);

			for (std::size_t i = 0; i < aMeshData.size(); i++) {
				if (!aMeshData[i].hasAlphaMask) continue;

				vkCmdBindDescriptorSets(
					aCmdBuff, 
					VK_PIPELINE_BIND_POINT_GRAPHICS, 
					aPipelineLayouts.regularPipelineLayout, 
					1, 
					1, 
					&aDescriptorSets.materialDescriptors[aMeshData[i].materialId], 
					0,
					nullptr
				);

				VkBuffer vbuffers[3] = {
					aMeshData[i].positionBuffer.buffer,
					aMeshData[i].texCoordBuffer.buffer,
					aMeshData[i].normalsBuffer.buffer
				};
				VkBuffer ibuffer = aMeshData[i].indicesBuffer.buffer;
				VkDeviceSize voffsets[3]{};
				VkDeviceSize ioffset{};

				vkCmdBindVertexBuffers(aCmdBuff, 0, 3, vbuffers, voffsets);
				vkCmdBindIndexBuffer(aCmdBuff, ibuffer, ioffset, VK_INDEX_TYPE_UINT32);

				vkCmdDrawIndexed(aCmdBuff, aMeshData[i].indicesCount, 1, 0, 0, 0);
			}

			vkCmdEndRenderPass(aCmdBuff);

			// Post processing render pass

			VkClearValue clearValues2[1]{};
			clearValues[0].color.float32[0] = 0.1f;
			clearValues[0].color.float32[1] = 0.1f;
			clearValues[0].color.float32[2] = 0.1f;
			clearValues[0].color.float32[3] = 1.0f;

			VkRenderPassBeginInfo passInfo2{};
			passInfo2.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			passInfo2.renderPass = aRenderPasses.postProcessRenderPass;
			passInfo2.framebuffer = aFramebuffers.fullscreenSwapchainFramebuffer;
			passInfo2.renderArea.offset = VkOffset2D{0, 0};
			passInfo2.renderArea.extent = aExtent;
			passInfo2.clearValueCount = 1;
			passInfo2.pClearValues = clearValues2;

			vkCmdBeginRenderPass(aCmdBuff, &passInfo2, VK_SUBPASS_CONTENTS_INLINE);

			vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aPipelines.postProcessPipeline);

			vkCmdBindDescriptorSets(
				aCmdBuff, 
				VK_PIPELINE_BIND_POINT_GRAPHICS, 
				aPipelineLayouts.postProcessPipelineLayout, 
				0, 
				1, 
				&aDescriptorSets.postProcessDescriptor, 
				0, 
				nullptr
			);

			vkCmdDraw(aCmdBuff, 3, 1, 0, 0);

			vkCmdEndRenderPass(aCmdBuff);
		}

		if (const auto res = vkEndCommandBuffer(aCmdBuff); VK_SUCCESS != res)
			throw lut::Error("Unable to end recording command buffer\n vkEndCommandBuffer() returned %s", lut::to_string(res).c_str());
	}

	void submit_commands(const lut::VulkanWindow& aWindow, VkCommandBuffer aCmdBuff, VkFence aFence, VkSemaphore aWaitSemaphore, VkSemaphore aSignalSemaphore) {
		VkPipelineStageFlags waitPipelineStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

		VkSubmitInfo subInfo{};
		subInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		subInfo.commandBufferCount = 1;
		subInfo.pCommandBuffers = &aCmdBuff;
		subInfo.waitSemaphoreCount = 1;
		subInfo.pWaitSemaphores = &aWaitSemaphore;
		subInfo.pWaitDstStageMask = &waitPipelineStages;
		subInfo.signalSemaphoreCount = 1;
		subInfo.pSignalSemaphores = &aSignalSemaphore;

		if (const auto res = vkQueueSubmit(aWindow.graphicsQueue, 1, &subInfo, aFence); VK_SUCCESS != res) {
			throw lut::Error("Unable to submit command buffer to queue\n vkQueueSubmit() returned %s", lut::to_string(res).c_str());
		}
	}
}


//EOF vim:syntax=cpp:foldmethod=marker:ts=4:noexpandtab: 
