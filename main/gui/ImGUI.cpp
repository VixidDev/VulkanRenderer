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

void GUI::draw() {
	Renderer& renderer = this->driver->getRenderer();

	ImGui::ShowDemoWindow();

	ImGui::Begin("Debug Menu");

	// Shadow Settings
	ImGui::Text("Shadow settings:");
	if (ImGui::Checkbox("Shadows", &renderer.getShadowsEnabled())) {
		renderer.setRecreateSwapchain(true, true);
	}

	ImGui::Checkbox("Shadow Map Texture", &this->showShadowMapTexture);
	ImGui::InputInt("Num of lights", &renderer.numLights);

	ImGui::Separator();

	ImGui::Checkbox("Sun View Debug", &this->showSunView);

	// Depth Bias Settings
	ImGui::Text("Depth bias settings:");
	ImGui::SliderFloat("Depth Bias Constant", &renderer.getDepthBiasConstant(), 0.0f, 10.0f);
	ImGui::SliderFloat("Depth Bias Slope Factor", &renderer.getDepthBiasSlopeFactor(), 0.0f, 10.0f);
	ImGui::SliderFloat("Shadow Bias", &renderer.shadowBias, 0.0f, 0.1f);

	// Camera debug
	Camera& camera = renderer.getCamera();

	ImGui::Text("Camera vars:");
	ImGui::Text("Pos: %f %f %f", camera.getPosition().x, camera.getPosition().y, camera.getPosition().z);
	ImGui::Text("Yaw: %f - Pitch %f", camera.getYaw(), camera.getPitch());

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
					ImGui::Image((ImTextureID)descriptorSet->getDescriptorSets()[this->pointLightShadowIndex], ImVec2(this->shadowMapSize[0], this->shadowMapSize[1]));
				}

				ImGui::EndTabItem();
			}
			if (ImGui::BeginTabItem("Directional Lights")) {
				if (ImGui::InputInt("Directional Light Shadow Map Index", &this->dirLightShadowIndex)) {
					this->dirLightShadowIndex = std::clamp(this->dirLightShadowIndex, 0, std::max(0, (int)renderer.numDirectionalLights - 1));
				}

				ArrayImageDescriptorSet* descriptorSet = dynamic_cast<ArrayImageDescriptorSet*>(renderer.getDescriptorSet("directionalLightShadowsDebug"));
				if (renderer.numDirectionalLights > 0 && descriptorSet) {
					ImGui::Image((ImTextureID)descriptorSet->getDescriptorSets()[this->dirLightShadowIndex], ImVec2(this->shadowMapSize[0], this->shadowMapSize[1]));
				}

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

		ImGui::Image((ImTextureID)renderer.getDescriptorSet("sunView")->getHandle(), ImVec2(this->sunViewSize[0], this->sunViewSize[1]));

		ImGui::End();
	}
}