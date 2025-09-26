#pragma once

#include <volk/volk.h>

class Driver;

class GUI {
public:
	GUI() = default;
	GUI(Driver* driver);

	~GUI() = default;

	void init(VkRenderPass guiRenderPass);
	void prepare();
private:
	Driver* driver = nullptr;

	bool showShadowMapTexture = false;
	bool showSunView = false;
	int shadowMapSize[2] = { 1500, 1500 };
	int sunViewSize[2] = { 1500, 1500 };

	int pointLightShadowIndex = 0;
	int dirLightShadowIndex = 0;
	int spotLightShadowIndex = 0;

	int selectedLight = 0;

	void draw();
};