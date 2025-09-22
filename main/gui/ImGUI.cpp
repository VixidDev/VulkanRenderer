#include "ImGUI.hpp"

#include <algorithm>

#include "../imgui/imgui.h"
#include "../imgui/backends/imgui_impl_glfw.h"
#include "../imgui/backends/imgui_impl_vulkan.h"

#include "../Driver.hpp"
#include "../rendering/Renderer.hpp"
#include "../vulkan/VulkanDevice.hpp"

#include "../rendering/objects/impl/descriptorSets/ArrayImageDescriptorSet.hpp"

GUI::GUI(Driver* driver) : driver(driver) {}

void GUI::init(VkRenderPass guiRenderPass) {
	IMGUI_CHECKVERSION();

	ImGuiContext* imGuiContext = ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

	VulkanWindow* window = this->driver->getRenderer().getContext().window.get();
	GLFWwindow* glfwWindow = window->window;

	ImGui_ImplGlfw_InitForVulkan(glfwWindow, true);

	ImGui_ImplVulkan_InitInfo initInfo{};
	initInfo.Instance = window->instance;
	initInfo.PhysicalDevice = window->physicalDevice;
	initInfo.Device = window->device->device;
	initInfo.QueueFamily = window->graphicsFamilyIndex;
	initInfo.Queue = window->graphicsQueue;
	initInfo.DescriptorPool = window->device->descPool;
	initInfo.RenderPass = guiRenderPass;
	initInfo.Subpass = 0;
	initInfo.MinImageCount = 2;
	initInfo.ImageCount = window->minImageCount;
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

	//ImGui::ShowDemoWindow();

	ImGui::Begin("Debug Menu");

	if (ImGui::BeginTabBar("Main Debug Menu")) {
		if (ImGui::BeginTabItem("Misc")) {
			// Shadow Settings
			ImGui::Text("Shadow settings:");
			if (ImGui::Checkbox("Shadows", &renderer.getShadowsEnabled())) {
				renderer.setRecreateSwapchain(true, true);
			}

			ImGui::Separator();

			// Depth Bias Settings
			ImGui::Text("Depth bias settings:");
			ImGui::SliderFloat("Depth Bias Constant", &renderer.getDepthBiasConstant(), 0.0f, 10.0f);
			ImGui::SliderFloat("Depth Bias Slope Factor", &renderer.getDepthBiasSlopeFactor(), 0.0f, 10.0f);

			ImGui::Separator();

			// Camera debug
			Camera& camera = renderer.getCamera();

			ImGui::Text("Camera Debug");
			ImGui::Text("Pos: %f %f %f", camera.getPosition().x, camera.getPosition().y, camera.getPosition().z);
			ImGui::Text("Yaw: %f - Pitch %f", camera.getYaw(), camera.getPitch());
			ImGui::SliderFloat("Camera FOV", &camera.getFov(), 1.0f, 145.0f);
			ImGui::SliderFloat("Camera Near Plane", &camera.getNearPlane(), 0.0001f, 1.0f, "%.5f");
			ImGui::SliderFloat("Camera Far Plane", &camera.getFarPlane(), 1.0f, 1024.0f);

			ImGui::EndTabItem();
		}
		if (ImGui::BeginTabItem("Lights")) {
			if (renderer.getDebugView())
				ImGui::BeginDisabled();

			ImGui::Checkbox("Shadow Map Texture", &this->showShadowMapTexture);
			ImGui::Checkbox("Sun View Debug", &this->showSunView);

			if (renderer.getDebugView())
				ImGui::EndDisabled();

			ImGui::Separator();

			ImGui::InputInt("Num of lights", &renderer.numLights);

			ImGui::Separator();

			ImGui::Text("Sun Light Debug");
			ImGui::SliderFloat("Ortho bounds", &renderer.sunOrthoBounds, 0.1f, 50.0f);
			ImGui::SliderFloat("Near plane", &renderer.sunShadowNear, 0.001f, 10.0f);
			ImGui::SliderFloat("Far plane", &renderer.sunShadowFar, 1.0f, 1024.0f);
			ImGui::SliderFloat("Sun distance", &renderer.sunDistance, 1.0f, 100.0f);

			ImGui::Separator();

			ImGui::Text("Light Editor");
			if (ImGui::InputInt("Light index", &this->selectedLight)) {
				this->selectedLight = std::clamp(this->selectedLight, 0, std::max(0, renderer.numLights - 1));
			}

			SSBOs& ssbos = renderer.getSSBOs();
			glsl::Light& light = ssbos.lights.at(this->selectedLight);
			ImGui::ColorEdit3("Colour", &light.colour[0]);
			ImGui::SliderInt("Intensity", &light.metadata.z, 1, 500);

			ImGui::EndTabItem();
		}
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
		ImGui::EndTabBar();
	}

	//ImGui::SliderFloat("zMult", &renderer.zMult, 0.1f, 100.0f);

	ImGui::End();

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

				//ArrayImageDescriptorSet* descriptorSet = dynamic_cast<ArrayImageDescriptorSet*>(renderer.getDescriptorSet("spotLightShadowsDebug"));
				//if (renderer.numSpotLights > 0 && descriptorSet) {
				//	ImGui::Image((ImTextureID)descriptorSet->getDescriptorSets()[this->spotLightShadowIndex], ImVec2(this->shadowMapSize[0], this->shadowMapSize[1]));
				//}

				ImGui::EndTabItem();
			}

			ImGui::EndTabBar();
		}

		ImGui::End();
	}

	// Debug Sun View Texture
	if (this->showSunView) {
		ImGui::Begin("Sun View Texture");

		ImGui::Checkbox("Show Camera Frustum Bounds", &renderer.renderCameraFrustumBounds);
		ImGui::InputInt2("Sun View Texture Size", this->sunViewSize);

		ImGui::Image((ImTextureID)renderer.getDescriptorSet("sunView")->getHandle(), ImVec2(static_cast<float>(this->sunViewSize[0]), static_cast<float>(this->sunViewSize[1])));

		ImGui::End();
	}
}