#pragma once

#define SUCCESS 1
#define FAIL 0

#include <chrono>

#include "rendering/Renderer.hpp"
#include "gui/ImGui.hpp"
#include "input/Callbacks.hpp"
#include "baked/BakedModel.hpp"
#include "rendering/lights/Light.hpp"
#include "vulkan/objects/VkObjects.hpp"
#include "vulkan/objects/VkImage.hpp"

using Clock = std::chrono::steady_clock;
using Timepoint = Clock::time_point;
using Seconds = std::chrono::duration<float, std::ratio<1>>;

class Driver {
public:
	Driver();
	~Driver() = default;

	int init();

	int loadScene();
	int uploadInitialToGPU();

	void run();

	Renderer& getRenderer();

	std::vector<std::pair<vk::Image, vk::ImageView>>& getSceneTextures();
	std::vector<VkDescriptorSet>& getMaterialDescriptors();
	std::vector<MeshData>& getMeshData();
private:
	Renderer renderer;
	GUI gui;
	UserState state{};

	BakedModel bakedModel;
	std::vector<std::pair<vk::Image, vk::ImageView>> sceneTextures;
	std::vector<VkDescriptorSet> materialDescriptors;
	std::vector<Light> lights{};
	std::vector<MeshData> meshData;
};