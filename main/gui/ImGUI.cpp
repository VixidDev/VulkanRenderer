#include "ImGUI.hpp"

#include "../imgui/imgui.h"
#include "../imgui/backends/imgui_impl_glfw.h"
#include "../imgui/backends/imgui_impl_vulkan.h"

#include "../Driver.hpp"
#include "../rendering/Renderer.hpp"
#include "../vulkan/VulkanDevice.hpp"

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

	ImGui::Begin("Debug Menu");

	// Shadow Settings
	ImGui::Text("Shadow settings:");
	if (ImGui::Checkbox("Shadows", &renderer.getShadowsEnabled())) {
		renderer.setRecreateSwapchain(true, true);
	}

	ImGui::Checkbox("Shadow Map Texture", &this->showShadowMapTexture);

	// Depth Bias Settings
	ImGui::Text("Depth bias settings:");
	ImGui::SliderFloat("Depth Bias Constant", &renderer.getDepthBiasConstant(), 0.0f, 10.0f);
	ImGui::SliderFloat("Depth Bias Slope Factor", &renderer.getDepthBiasSlopeFactor(), 0.0f, 10.0f);

	// Camera debug
	Camera& camera = renderer.getCamera();

	ImGui::Text("Camera vars:");
	ImGui::Text("Yaw: %f - Pitch %f", camera.getYaw(), camera.getPitch());

	ImGui::End();

	// Debug Shadow Map Texture
	if (this->showShadowMapTexture) {
		ImGui::Begin("Shadow Map Texture");
		ImGui::InputInt2("Shadow Map Texture Size", this->shadowMapSize);
		//ImGui::Checkbox("Control Shadow Casting Light", );
		ImGui::Image((ImTextureID)renderer.getDescriptorSet("debugLinearDepth")->getHandle(), ImVec2(this->shadowMapSize[0], this->shadowMapSize[1]));
		ImGui::End();
	}
}