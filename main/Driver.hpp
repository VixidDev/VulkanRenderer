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
#include "rendering/management/TimestampManager.hpp"

using Clock = std::chrono::steady_clock;
using Timepoint = Clock::time_point;
using Seconds = std::chrono::duration<float, std::ratio<1>>;

class Driver {
public:
	Driver();
	~Driver() = default;

	Driver(const Driver&) = delete;
	Driver& operator=(const Driver&) = delete;
	Driver(Driver&&) = delete;
	Driver* operator=(Driver&&) = delete;

	int init();

	int loadScene();
	int uploadInitialToGPU();

	void run();

	Renderer& getRenderer() { return this->renderer; }
	TimestampManager& getTimestampManager() { return this->timestampManager; }
	const float getTimeDelta() const { return this->timeDelta; }

	std::vector<std::pair<vk::Image, vk::ImageView>>& getSceneTextures() { return this->sceneTextures; }
	std::vector<VkDescriptorSet>& getMaterialDescriptors() { return this->materialDescriptors; }
	std::vector<Light>& getLights() { return this->lights; }
	std::vector<MeshData>& getMeshData() { return this->meshData; }
private:
	Renderer renderer;
	TimestampManager timestampManager;
	GUI gui;
	UserState state{};

	float timeDelta{};

	BakedModel bakedModel;
	std::vector<std::pair<vk::Image, vk::ImageView>> sceneTextures;
	std::vector<VkDescriptorSet> materialDescriptors;
	std::vector<Light> lights{};
	std::vector<MeshData> meshData;
};