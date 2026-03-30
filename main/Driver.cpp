#include "Driver.hpp"

#include <iostream>

#include "input/Callbacks.hpp"
#include "baked/BakedModelLoader.hpp"
#include "rendering/lights/Lights.hpp"

Driver::Driver() : renderer(this) {};

int Driver::init() {
	VulkanWindow& window = *this->renderer.getContext().window;

	this->timestampManager = TimestampManager(&this->renderer.getContext());

	// Set GLFW user pointer
	glfwSetWindowUserPointer(window.getGLFWwindow(), &this->state);

	// Set GLFW callbacks
	glfwSetKeyCallback(window.getGLFWwindow(), &Callbacks::onKey);
	glfwSetMouseButtonCallback(window.getGLFWwindow(), &Callbacks::onMouseButton);
	glfwSetCursorPosCallback(window.getGLFWwindow(), &Callbacks::onMouseMove);

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

	const Timepoint beforeLoadBakedModel = Clock::now();
	this->bakedModel = loadBakedModel("assets/main/suntemple.vixidvkmesh");
	const Timepoint afterLoadBakedModel = Clock::now();
	float bakedModel = std::chrono::duration_cast<Seconds>(afterLoadBakedModel - beforeLoadBakedModel).count();
	std::fprintf(stderr, "loadBakedModel took: %.3f ms\n", bakedModel * 1000.0f);

	const Timepoint beforeLoadTextures = Clock::now();
	this->sceneTextures = BakedModelLoader::loadTextures(context, this->bakedModel);
	const Timepoint afterLoadTextures = Clock::now();
	float loadTextures = std::chrono::duration_cast<Seconds>(afterLoadTextures - beforeLoadTextures).count();
	std::fprintf(stderr, "loadTextures took: %.3f ms\n", loadTextures * 1000.0f);

	// TODO: Update to use similar objects like in Renderer.cpp
	this->materialDescriptors = BakedModelLoader::createMaterialDescriptors(*this, this->bakedModel);

	const std::string lightsFile = "assets-src/main/lights.vl";
	if (!Lights::parse(lightsFile)) {
		std::fprintf(stderr, "Failed to parse lights for file '%s'\n", lightsFile.c_str());
	}

	this->renderer.setLights(&this->lights);

	return SUCCESS;
}

int Driver::uploadInitialToGPU() {
	VulkanContext& context = this->renderer.getContext();
	
	this->meshData = BakedModelLoader::uploadToGPU(context, this->bakedModel);

	return SUCCESS;
}

void Driver::run() {
	VulkanWindow& window = *this->renderer.getContext().window;

	Timepoint previous = Clock::now();

	while (!glfwWindowShouldClose(window.getGLFWwindow())) {
		// Calculate time delta
		const Timepoint now = Clock::now();
		this->timeDelta = std::chrono::duration_cast<Seconds>(now - previous).count();
		previous = now;

		this->timestampManager.writeCPUTimestamp("entireFrame");

		// Poll IO events
		glfwPollEvents();

		this->gui.calculateFPS(this->timeDelta);
		this->timestampManager.writeCPUTimestamp("guiPrepare");
		this->gui.prepare();
		this->timestampManager.writeCPUTimestamp("guiPrepare");

		if (this->renderer.checkSwapchain()) {
			this->timestampManager.clearCPUTimestamps();
			continue;
		}

		if (this->renderer.acquireSwapchainImage()) {
			this->timestampManager.clearCPUTimestamps();
			continue;
		}

		this->timestampManager.writeCPUTimestamp("renderUpdate");
		this->renderer.update(this->timeDelta);
		this->timestampManager.writeCPUTimestamp("renderUpdate");

		this->timestampManager.writeCPUTimestamp("render");
		this->renderer.render();
		this->timestampManager.writeCPUTimestamp("render");

		this->timestampManager.writeCPUTimestamp("submitRender");
		this->renderer.submitRender();
		this->timestampManager.writeCPUTimestamp("submitRender");

		this->timestampManager.writeCPUTimestamp("entireFrame");
	
		this->timestampManager.flushCPUTimestamps();
	}

	this->renderer.finishRendering();
}
