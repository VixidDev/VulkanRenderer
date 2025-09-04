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
	int shadowMapSize[2] = { 1500, 1500 };

	void draw();
};