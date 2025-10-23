#include "ImGUI.hpp"

#include <algorithm>
#include <numeric>

#include "../imgui/imgui.h"
#include "../imgui/backends/imgui_impl_glfw.h"
#include "../imgui/backends/imgui_impl_vulkan.h"

#include "../Driver.hpp"
#include "../rendering/Renderer.hpp"
#include "../vulkan/VulkanDevice.hpp"
#include "../vulkan/Swapchain.hpp"

// Concrete types
#include "../rendering/objects/impl/descriptorSets/ArrayImageDescriptorSet.hpp"
#include "../rendering/postProcessing/TonemapPostProcess.hpp"

GUI::GUI(Driver* driver) : driver(driver) {
	this->frameTimes.reserve(1000);
}

void GUI::init(VkRenderPass guiRenderPass) {
	IMGUI_CHECKVERSION();

	ImGuiContext* imGuiContext = ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

	VulkanWindow* window = this->driver->getRenderer().getContext().window.get();
	VulkanDevice* device = window->getDevice();
	GLFWwindow* glfwWindow = window->getGLFWwindow();

	ImGui_ImplGlfw_InitForVulkan(glfwWindow, true);

	ImGui_ImplVulkan_InitInfo initInfo{};
	initInfo.Instance = window->getInstance();
	initInfo.PhysicalDevice = device->getPhysicalDevice();
	initInfo.Device = device->getDevice();
	initInfo.QueueFamily = device->getGraphicsFamilyIndex();
	initInfo.Queue = device->getGraphicsQueue();
	initInfo.DescriptorPool = device->getDescPool();
	initInfo.RenderPass = guiRenderPass;
	initInfo.Subpass = 0;
	initInfo.MinImageCount = 2;
	initInfo.ImageCount = window->getSwapchain()->getMinImageCount();
	initInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
	initInfo.CheckVkResultFn = [](VkResult err) {
		if (err != VK_SUCCESS)
			std::printf("[ImGui] Vulkan error: %d\n", err);
	};

	ImGui_ImplVulkan_Init(&initInfo);
}

void GUI::prepare() {
	ImGui_ImplVulkan_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	this->draw();

	ImGui::EndFrame();
	ImGui::Render();
}

// Taken from imgui_demo.cpp
static void HelpMarker(const char* desc) {
	ImGui::TextDisabled("(?)");
	if (ImGui::BeginItemTooltip()) {
		ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
		ImGui::TextUnformatted(desc);
		ImGui::PopTextWrapPos();
		ImGui::EndTooltip();
	}
}

void GUI::draw() {
	Renderer& renderer = this->driver->getRenderer();

	ImGui::ShowDemoWindow();

	SSAOPreProcess* ssaoPPE = renderer.getSSAOPreProcess();

	// I dont like this, but it is what it is
	auto& bloomPPE = renderer.getPostProcessingEffects()[0].second;
	auto& tonemapPPE = renderer.getPostProcessingEffects()[1].second;
	auto& fxaaPPE = renderer.getPostProcessingEffects()[2].second;
	auto& mosaicPEE = renderer.getPostProcessingEffects()[3].second;

	ImGui::Begin("Debug Menu");

	if (ImGui::BeginTabBar("Main Debug Menu")) {
		if (ImGui::BeginTabItem("Main")) {
			// Present mode
			ImGui::Text("Present Mode:");
			
			const std::vector<std::string>& presentModeStrings = renderer.getContext().window->getSwapchain()->getPresentModeStrings();
			int& presentMode = renderer.getContext().window->getSwapchain()->getPresentMode();
			for (std::size_t i = 0; i < presentModeStrings.size(); i++) {
				if (ImGui::RadioButton(presentModeStrings[i].c_str(), &presentMode, i)) {
					renderer.setRecreateSwapchain(true, true);
				}
			}

			ImGui::Separator();

			// Renderer type
			ImGui::Text("Rendering type:");
			ImGui::RadioButton("Forward", &renderer.getRenderingType(), 0); ImGui::SameLine();
			ImGui::RadioButton("Deferred", &renderer.getRenderingType(), 1);

			ImGui::Separator();

			// Shadow Settings
			ImGui::Text("Shadow settings:");
			if (ImGui::Checkbox("Shadows", &renderer.getShadowsEnabled())) {
				renderer.setRecreateSwapchain(true, true);
			}
			if (renderer.getShadowsEnabled()) {
				int oldVSMState = renderer.vsmShadowsEnabled;

				ImGui::RadioButton("Standard", &renderer.vsmShadowsEnabled, 0); ImGui::SameLine();
				ImGui::RadioButton("VSM", &renderer.vsmShadowsEnabled, 1);

				if (oldVSMState != renderer.vsmShadowsEnabled)
					renderer.setRecreateSwapchain(true, true);
			}

			ImGui::Separator();

			// Depth Bias Settings
			ImGui::Text("Depth bias settings:");
			ImGui::SliderFloat("Depth Bias Constant", &renderer.getDepthBiasConstant(), 0.0f, 10.0f);
			ImGui::SliderFloat("Depth Bias Slope Factor", &renderer.getDepthBiasSlopeFactor(), 0.0f, 10.0f);

			ImGui::Separator();

			// Camera debug
			Camera* camera = renderer.getCamera();

			ImGui::Text("Camera Debug");
			ImGui::Text("Pos: %f %f %f", camera->getPosition().x, camera->getPosition().y, camera->getPosition().z);
			ImGui::Text("Yaw: %f - Pitch %f", camera->getYaw(), camera->getPitch());
			if (ImGui::SliderFloat("Camera FOV", &camera->getFov(), 1.0f, 145.0f)) {
				camera->markProjectionDirty();
			}
			if (ImGui::SliderFloat("Camera Near Plane", &camera->getNearPlane(), 0.0001f, 1.0f, "%.5f")) {
				camera->markProjectionDirty();
			}
			if (ImGui::SliderFloat("Camera Far Plane", &camera->getFarPlane(), 1.0f, 1024.0f)) {
				camera->markProjectionDirty();
			}

			ImGui::EndTabItem();
		}
		if (ImGui::BeginTabItem("Lights")) {
#ifndef NDEBUG
			if (renderer.getDebugView())
				ImGui::BeginDisabled();
			if (!renderer.getShadowsEnabled())
				ImGui::BeginDisabled();

			ImGui::Checkbox("Shadow Map Texture", &this->showShadowMapTexture);

			if (!renderer.getShadowsEnabled())
				ImGui::EndDisabled();

			if (renderer.getRenderingType())
				ImGui::BeginDisabled();

			ImGui::Checkbox("Sun View Debug", &renderer.showSunView);

			if (renderer.getRenderingType())
				ImGui::EndDisabled();
			if (renderer.getDebugView())
				ImGui::EndDisabled();
#endif

			ImGui::Separator();

			ImGui::InputInt("Num of lights", &renderer.numLights);

			ImGui::Separator();

			ImGui::SliderFloat("Emissive Strength", &renderer.emissiveStrength, 1.0f, 100.0f);
			ImGui::SliderFloat("Shadow Bias", &renderer.shadowBias, 0.0001f, 0.01f, "%.5f");
			ImGui::SliderFloat("Bleed Reduction", &renderer.bleedReduction, 0.0f, 1.0f, "%.5f");

			int oldVsmTapSize = renderer.vsmTapSize;

			ImGui::Text("VSM Tap Size:"); ImGui::SameLine();
			ImGui::RadioButton("5x5", &renderer.vsmTapSize, TapSize::e5X5); ImGui::SameLine();
			ImGui::RadioButton("9x9", &renderer.vsmTapSize, TapSize::e9X9); ImGui::SameLine();
			ImGui::RadioButton("17x17", &renderer.vsmTapSize, TapSize::e17X17); ImGui::SameLine();
			ImGui::RadioButton("25x25", &renderer.vsmTapSize, TapSize::e25X25); ImGui::SameLine();
			ImGui::RadioButton("41x41", &renderer.vsmTapSize, TapSize::e41X41);

			if (oldVsmTapSize != renderer.vsmTapSize)
				renderer.setRecreateSwapchain(true, true); // Need to recreate since tap sizes are a specialization constant

			ImGui::Separator();

			ImGui::Text("Shadow Map Resolutions");
			const char* shadowResolutions[4] = { "1024x1024", "2048x2048", "4096x4096", "8192x8192" };

			if (ImGui::Combo("Sun Shadow Map Resolution", &renderer.sunShadowMapResIdx, shadowResolutions, 4)) {
				renderer.setRecreateSwapchain(true, true);
			}

			if (ImGui::Combo("Point Shadow Map Resolution", &renderer.pointShadowMapResIdx, shadowResolutions, 4)) {
				renderer.setRecreateSwapchain(true, true);
			}

			ImGui::Separator();

			ImGui::Text("Sun Params");
			ImGui::SliderFloat("upperStep", &renderer.sunUpperStep, 0.01f, 1.0f);
			ImGui::SliderFloat("lowerStep", &renderer.sunLowerStep, 0.01f, 1.0f);
			ImGui::SliderFloat("intensity", &renderer.sunIntensity, 1.0f, 10.0f);

			ImGui::Separator();

			std::vector<Light>& lights = this->driver->getLights();
			Light& sunLight = lights[renderer.getSunLightIndex()];

			ImGui::Text("Sun Light Debug");
			if (ImGui::SliderFloat("Ortho bounds", &renderer.sunOrthoBounds, 0.1f, 50.0f)) {
				renderer.getSunMatrices().projection.markDirty();
				sunLight.markDirty();
			}
			if (ImGui::SliderFloat("Near plane", &renderer.sunShadowNear, 0.001f, 10.0f)) {
				renderer.getSunMatrices().projection.markDirty();
				sunLight.markDirty();
			}
			if (ImGui::SliderFloat("Far plane", &renderer.sunShadowFar, 1.0f, 1024.0f)) {
				renderer.getSunMatrices().projection.markDirty();
				sunLight.markDirty();
			}
			if (ImGui::SliderFloat("Sun distance", &renderer.sunDistance, 1.0f, 100.0f)) {
				renderer.getSunMatrices().view.markDirty();
				sunLight.markDirty();
			}

			ImGui::Separator();

			ImGui::Text("Light Editor");
			if (ImGui::InputInt("Light index", &this->selectedLight)) {
				this->selectedLight = std::clamp(this->selectedLight, 0, std::max(0, renderer.numLights - 1));
			}

			SSBOs& ssbos = renderer.getSSBOs();
			glsl::Light& light = ssbos.lights.at(this->selectedLight);
			ImGui::ColorEdit3("Colour", &light.colourAndIntensity[0]);
			ImGui::SliderFloat("Intensity", &light.colourAndIntensity.w, 1.0f, 100.0f);

			ImGui::EndTabItem();
		}
		if (ImGui::BeginTabItem("Effects")) {
			ImGui::Text("Pre Processing Effects");
			if (ImGui::Checkbox("SSAO", &renderer.ssaoEnabled)) {
				renderer.setRecreateSwapchain(true, true);
			}
			if (renderer.ssaoEnabled) {
				ImGui::Text("Main SSAO Pass");
				ImGui::SliderFloat("Radius", &renderer.getUniforms().ssaoUniform.radius, 0.1f, 10.0f);
				ImGui::Text("SSAO Blur Pass");
				ImGui::SliderInt("Blur radius", &ssaoPPE->blurPC.radius, 1, 5);
				ImGui::SliderFloat("Depth threshold", &ssaoPPE->blurPC.depthThreshold, 0.0f, 0.1f, "%.5f");
				ImGui::SliderFloat("Normal threshold", &ssaoPPE->blurPC.normalThreshold, 0.0f, 1.0f);
				ImGui::Text("Applying SSAO Pass");
				ImGui::SliderFloat("Exp", &renderer.ssaoExp, 1.0f, 5.0f);
			}

			ImGui::Separator();

			ImGui::Text("Post Processing Effects");
			ImGui::Checkbox("Bloom", &bloomPPE->getEnabled());
			if (bloomPPE->getEnabled()) {
				int oldBloomTapSize = renderer.bloomTapSize;

				ImGui::Text("Tap Sizes:"); ImGui::SameLine();
				ImGui::RadioButton("5x5", &renderer.bloomTapSize, TapSize::e5X5); ImGui::SameLine();
				ImGui::RadioButton("9x9", &renderer.bloomTapSize, TapSize::e9X9); ImGui::SameLine();
				ImGui::RadioButton("17x17", &renderer.bloomTapSize, TapSize::e17X17); ImGui::SameLine();
				ImGui::RadioButton("25x25", &renderer.bloomTapSize, TapSize::e25X25); ImGui::SameLine();
				ImGui::RadioButton("41x41", &renderer.bloomTapSize, TapSize::e41X41);

				if (oldBloomTapSize != renderer.bloomTapSize)
					renderer.setRecreateSwapchain(true, true); // Need to recreate since tap sizes are a specialization constant

				ImGui::SliderInt("Blur Iterations", &renderer.bloomIterations, 1, 10);
				ImGui::SliderFloat("Threshold", &renderer.brightnessThreshold, 0.0f, 1.0f);
			}

			ImGui::BeginDisabled();
			ImGui::Checkbox("Tonemap", &tonemapPPE->getEnabled());
			ImGui::EndDisabled();
			if (tonemapPPE->getEnabled()) {
				TonemapPostProcess* tonemapImpl = dynamic_cast<TonemapPostProcess*>(tonemapPPE.get());
				ImGui::RadioButton("Just Gamma", &tonemapImpl->getTonemap(), Tonemap::JUST_GAMMA); ImGui::SameLine(); HelpMarker("Only applies gamma correction. (Ignores exposure)");
				ImGui::RadioButton("Filmic", &tonemapImpl->getTonemap(), Tonemap::FILMIC);
				ImGui::RadioButton("Uncharted", &tonemapImpl->getTonemap(), Tonemap::UNCHARTED);
				ImGui::RadioButton("ACES", &tonemapImpl->getTonemap(), Tonemap::ACES);
				ImGui::RadioButton("AgX", &tonemapImpl->getTonemap(), Tonemap::AGX);
				ImGui::RadioButton("Khronos PBR", &tonemapImpl->getTonemap(), Tonemap::KHRONOS_PBR);
				ImGui::SliderFloat("Exposure", &tonemapImpl->getExposure(), 0.0f, 5.0f);
			}

			ImGui::Checkbox("FXAA", &fxaaPPE->getEnabled());
			ImGui::Checkbox("Mosaic", &mosaicPEE->getEnabled());

			ImGui::EndTabItem();
		}
#ifndef NDEBUG
		if (ImGui::BeginTabItem("Debug")) {
			ImGui::Checkbox("Enable Debug View", &renderer.getDebugView());

			if (renderer.getDebugView()) {
				ImGui::RadioButton("Show Normals", &renderer.getDebugState(), 0);
				ImGui::RadioButton("Show Mipmap Levels", &renderer.getDebugState(), 1);
				ImGui::RadioButton("Show Linear Depth", &renderer.getDebugState(), 2);
				ImGui::RadioButton("Show Partial Derivatives", &renderer.getDebugState(), 3);
				ImGui::RadioButton("Show Overdraw", &renderer.getDebugState(), 7);
				ImGui::RadioButton("Show Overshading", &renderer.getDebugState(), 8);
				ImGui::Text("PBR Debug"); ImGui::SameLine(); HelpMarker("Some of the PBR debug views will appear overexposed when multiple lights are active, they only really serve to show if the selected PBR function is working");
				ImGui::RadioButton("Show Distribution Function", &renderer.getDebugState(), 4);
				ImGui::RadioButton("Show Geometry Function", &renderer.getDebugState(), 5);
				ImGui::RadioButton("Show Fresnel Function", &renderer.getDebugState(), 6);
			}

			ImGui::EndTabItem();
		}
#endif
		ImGui::EndTabBar();
	}

	ImGui::End();

	// Performance Window
	ImGui::Begin("Performance");

	float frameTime = this->driver->getTimeDelta();
	ImGui::Text("FPS: %d - Avg FPS: %d", static_cast<int>(1 / frameTime), this->avgFps);
	ImGui::Text("Frame Time: %.3f ms - Avg Frame Time: %.3f ms", frameTime * 1000.0f, this->avgFrameTime * 1000.0f);

	TimestampManager& timestampManager = this->driver->getTimestampManager();
	TimestampReferences& cpuTimestampReferences = timestampManager.getCPUTimestampReferences();

	this->calculateAvgCpuTimestamps();

	ImGui::Text("CPU Times:");
	if (ImGui::BeginTable("cpuTimes", 3)) {
		ImGui::TableSetupColumn("Pass");
		ImGui::TableSetupColumn("Time took (ms)");
		ImGui::TableSetupColumn("Time took (ms) (1s avg)");
		ImGui::TableHeadersRow();

		for (std::size_t i = 0; i < cpuTimestampReferences.size(); i++) {
			const auto& [name, indices] = cpuTimestampReferences[i];
			std::uint64_t start = timestampManager.getCPUTimestamp(indices.start).value_or(0);
			std::uint64_t end   = timestampManager.getCPUTimestamp(indices.end).value_or(0);
			
			float timeTaken = static_cast<float>(end - start) / 1000000.0f; // Convert nanoseconds to milliseconds
			float avgTimeTaken = 0.0f;

			if (this->avgCpuTimeToReport.contains(name)) {
				avgTimeTaken = this->avgCpuTimeToReport[name];
			} else {
				avgTimeTaken = static_cast<float>(this->avgCpuTimes[name].first / std::max(1, this->avgCpuTimes[name].second)) / 1000000.0f;
				this->avgCpuTimeToReport.emplace(name, avgTimeTaken);
			}

			ImGui::TableNextRow();
			ImGui::TableSetColumnIndex(0);
			ImGui::Text("%s", name.c_str());
			ImGui::TableSetColumnIndex(1);
			ImGui::Text("%.3f", timeTaken);
			ImGui::TableSetColumnIndex(2);
			ImGui::Text("%.3f", avgTimeTaken);
		}

		ImGui::EndTable();
	}

	ImGui::Separator();

	TimestampReferences& gpuTimestampReferences = timestampManager.getGPUTimestampReferences();
	// Get timestamp period
	float timestampPeriod = renderer.getContext().window->getDevice()->getDeviceProperties().limits.timestampPeriod;

	this->calculateAvgGpuTimestamps();

	ImGui::Text("GPU Times:");
	if (ImGui::BeginTable("gpuTimes", 3)) {
		ImGui::TableSetupColumn("Pass");
		ImGui::TableSetupColumn("Time took (ms)");
		ImGui::TableSetupColumn("Time took (ms) (1s avg)");
		ImGui::TableHeadersRow();

		for (std::size_t i = 0; i < gpuTimestampReferences.size(); i++) {
			const auto& [name, indices] = gpuTimestampReferences[i];
			std::uint64_t start = timestampManager.getGPUTimestamp(indices.start).value_or(0);
			std::uint64_t end = timestampManager.getGPUTimestamp(indices.end).value_or(0);

			float timeTaken = static_cast<float>(end - start) * timestampPeriod / 1000000.0f; // Convert nanoseconds to milliseconds
			float avgTimeTaken = 0.0f;

			if (this->avgGpuTimeToReport.contains(name)) {
				avgTimeTaken = this->avgGpuTimeToReport[name];
			} else {
				avgTimeTaken = static_cast<float>(this->avgGpuTimes[name].first / std::max(1, this->avgGpuTimes[name].second)) * timestampPeriod / 1000000.0f;
				this->avgGpuTimeToReport.emplace(name, avgTimeTaken);
			}

			ImGui::TableNextRow();
			ImGui::TableSetColumnIndex(0);
			ImGui::Text("%s", name.c_str());
			ImGui::TableSetColumnIndex(1);
			ImGui::Text("%.3f", timeTaken);
			ImGui::TableSetColumnIndex(2);
			ImGui::Text("%.3f", avgTimeTaken);
		}

		ImGui::EndTable();
	}

	ImGui::End();
	
	if (!renderer.getShadowsEnabled()) {
		this->showShadowMapTexture = false;
	}
	// Debug Shadow Map Texture
	if (this->showShadowMapTexture) {
		
		ImGui::Begin("Shadow Map Texture");

		ImGui::InputInt2("Shadow Map Texture Size", this->shadowMapSize);

		if (ImGui::BeginTabBar("Shadow Map Tetxtures")) {
			if (ImGui::BeginTabItem("Point Lights")) {
				if (ImGui::InputInt("Point Light Shadow Map Index", &this->pointLightShadowIndex)) {
					this->pointLightShadowIndex = std::clamp(this->pointLightShadowIndex, 0, std::max(0, ((int)renderer.numPointLights * 6) - 1));
				}

				ArrayImageDescriptorSet* descriptorSet = dynamic_cast<ArrayImageDescriptorSet*>(renderer.getDescriptorSet("pointLightShadowsDebug"));
				if (renderer.numPointLights > 0 && descriptorSet) {
					ImGui::Image((ImTextureID)descriptorSet->getDescriptorSets()[this->pointLightShadowIndex], ImVec2(static_cast<float>(this->shadowMapSize[0]), static_cast<float>(this->shadowMapSize[1])));
				}

				ImGui::EndTabItem();
			}
			if (ImGui::BeginTabItem("Directional Light")) {
				ImGui::Image((ImTextureID)renderer.getDescriptorSet("directionalLightShadowDebug")->getHandle(), ImVec2(static_cast<float>(this->shadowMapSize[0]), static_cast<float>(this->shadowMapSize[1])));

				ImGui::EndTabItem();
			}
			if (ImGui::BeginTabItem("Spot Lights")) {
				if (ImGui::InputInt("Spot Light Shadow Map Index", &this->spotLightShadowIndex)) {
					this->spotLightShadowIndex = std::clamp(this->spotLightShadowIndex, 0, std::max(0, (int)renderer.numSpotLights - 1));
				}

				ArrayImageDescriptorSet* descriptorSet = dynamic_cast<ArrayImageDescriptorSet*>(renderer.getDescriptorSet("spotLightShadowsDebug"));
				if (renderer.numSpotLights > 0 && descriptorSet) {
					ImGui::Image((ImTextureID)descriptorSet->getDescriptorSets()[this->spotLightShadowIndex], ImVec2(this->shadowMapSize[0], this->shadowMapSize[1]));
				}

				ImGui::EndTabItem();
			}

			ImGui::EndTabBar();
		}

		ImGui::End();
	}

	// Debug Sun View Texture
	if (renderer.showSunView) {
		ImGui::Begin("Sun View Texture");

		ImGui::Checkbox("Show Camera Frustum Bounds", &renderer.renderCameraFrustumBounds);
		ImGui::InputInt2("Sun View Texture Size", this->sunViewSize);

		ImGui::Image((ImTextureID)renderer.getDescriptorSet("sunView")->getHandle(), ImVec2(static_cast<float>(this->sunViewSize[0]), static_cast<float>(this->sunViewSize[1])));

		ImGui::End();
	}
	if (this->secondTimer <= 0.0f) {
		this->avgCpuTimes.clear();
		this->avgGpuTimes.clear();
		this->avgCpuTimeToReport.clear();
		this->avgGpuTimeToReport.clear();
		this->secondTimer = 1.0f;
	}
}

void GUI::calculateAvgCpuTimestamps() {
	TimestampManager& timestampManager = this->driver->getTimestampManager();
	TimestampReferences& cpuTimestampReferences = timestampManager.getCPUTimestampReferences();

	for (const auto& [name, indices] : cpuTimestampReferences) {
		std::uint64_t start = timestampManager.getCPUTimestamp(indices.start).value_or(0);
		std::uint64_t end = timestampManager.getCPUTimestamp(indices.end).value_or(0);

		if (this->avgCpuTimes.contains(name)) {
			this->avgCpuTimes[name].first += end - start;
			this->avgCpuTimes[name].second++;
		} else {
			this->avgCpuTimes.emplace(name, std::make_pair(end - start, 1));
		}
	}
}

void GUI::calculateAvgGpuTimestamps() {
	TimestampManager& timestampManager = this->driver->getTimestampManager();
	TimestampReferences& gpuTimestampReferences = timestampManager.getGPUTimestampReferences();

	for (const auto& [name, indices] : gpuTimestampReferences) {
		std::uint64_t start = timestampManager.getGPUTimestamp(indices.start).value_or(0);
		std::uint64_t end = timestampManager.getGPUTimestamp(indices.end).value_or(0);

		if (this->avgGpuTimes.contains(name)) {
			this->avgGpuTimes[name].first += end - start;
			this->avgGpuTimes[name].second++;
		} else {
			this->avgGpuTimes.emplace(name, std::make_pair(end - start, 1));
		}
	}
}

void GUI::calculateFPS(float timeDelta) {
	this->frames++;
	this->frameTimes.emplace_back(timeDelta);

	this->secondTimer -= timeDelta;
	if (this->secondTimer <= 0.0f) {
		this->avgFrameTime = std::reduce(this->frameTimes.begin(), this->frameTimes.end()) / this->frames;
		this->avgFps = 1 / this->avgFrameTime;
		this->frameTimes.clear();
		this->frames = 0;
	}
}
