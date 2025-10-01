#include "Driver.hpp"

#include <iostream>

#include "input/Callbacks.hpp"
#include "baked/BakedModelLoader.hpp"
#include "rendering/lights/LightsParser.hpp"

Driver::Driver() : renderer(this) {};

int Driver::init() {
	VulkanWindow& window = *this->renderer.getContext().window;

	this->timestampManager = TimestampManager(&this->renderer.getContext());

	// Set GLFW user pointer
	glfwSetWindowUserPointer(window.window, &this->state);

	// Set GLFW callbacks
	glfwSetKeyCallback(window.window, &Callbacks::onKey);
	glfwSetMouseButtonCallback(window.window, &Callbacks::onMouseButton);
	glfwSetCursorPosCallback(window.window, &Callbacks::onMouseMove);

	// Init GUI
	this->gui = GUI(this);
	this->gui.init(this->renderer.getRenderPass("gui")->getRenderPassHandle());

	// Load mesh data
	if (!this->loadScene()) {
		std::fprintf(stderr, "Failed to load scene data!\n");
		return FAIL;
	}

	// Upload mesh data to GPU
	if (!this->uploadInitialToGPU()) {
		std::fprintf(stderr, "Failed to upload initial scene data to GPU!\n");
		return FAIL;
	}

	return SUCCESS;
}

int Driver::loadScene() {
	VulkanContext& context = this->renderer.getContext();

	this->bakedModel = loadBakedModel("assets/main/suntemple.vixidvkmesh");

	this->sceneTextures = BakedModelLoader::loadTextures(context, this->bakedModel);

	// TODO: Update to use similar objects like in Renderer.cpp
	this->materialDescriptors = BakedModelLoader::createMaterialDescriptors(*this, this->bakedModel);

	const std::string lightsFile = "assets/main/lights.vl";
	if (!LightsParser::parseLights(lightsFile, this->lights)) {
		std::fprintf(stderr, "Failed to parse lights for file '%s'\n", lightsFile.c_str());
	}

	this->renderer.setLights(&this->lights);

	return SUCCESS;
}

int Driver::uploadInitialToGPU() {
	VulkanContext& context = this->renderer.getContext();
	
	this->meshData = BakedModelLoader::uploadToGPU(context, bakedModel);

	return SUCCESS;
}

void Driver::run() {
	VulkanWindow& window = *this->renderer.getContext().window;

	Timepoint previous = Clock::now();

	while (!glfwWindowShouldClose(window.window)) {
		// Calculate time delta
		const Timepoint now = Clock::now();
		const float timeDelta = std::chrono::duration_cast<Seconds>(now - previous).count();
		previous = now;

		// Poll IO events
		glfwPollEvents();

		this->gui.prepare();
	
		if (this->renderer.checkSwapchain())
			continue;

		if (this->renderer.acquireSwapchainImage())
			continue;
		
		this->renderer.update(timeDelta);
		this->renderer.render();
		this->renderer.submitRender();
	}

	this->renderer.finishRendering();
}

Renderer& Driver::getRenderer() {
	return this->renderer;
}

TimestampManager& Driver::getTimestampManager() {
	return this->timestampManager;
}

std::vector<std::pair<vk::Image, vk::ImageView>>& Driver::getSceneTextures() {
	return this->sceneTextures;
}

std::vector<VkDescriptorSet>& Driver::getMaterialDescriptors() {
	return this->materialDescriptors;
}

std::vector<MeshData>& Driver::getMeshData() {
	return this->meshData;
}
