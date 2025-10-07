#pragma once

#include <vector>
#include <string>
#include <map>

#include <volk/volk.h>

class Driver;

using AvgTimes = std::map<std::string, std::pair<std::uint64_t, int>>;

class GUI {
public:
	GUI() = default;
	GUI(Driver* driver);

	~GUI() = default;

	void init(VkRenderPass guiRenderPass);
	void prepare();

	void calculateFPS(float timeDelta);
private:
	Driver* driver = nullptr;

	bool showShadowMapTexture = false;
	int shadowMapSize[2] = { 1500, 1500 };
	int sunViewSize[2] = { 1500, 1500 };

	int pointLightShadowIndex = 0;
	int dirLightShadowIndex = 0;
	int spotLightShadowIndex = 0;

	int selectedLight = 0;

	AvgTimes avgCpuTimes;
	AvgTimes avgGpuTimes;
	std::map<std::string, float> avgCpuTimeToReport;
	std::map<std::string, float> avgGpuTimeToReport;

	int frames = 0;
	int avgFps = 0;
	float avgFrameTime = 0.0f, secondTimer = 1.0f;
	std::vector<float> frameTimes;

	void draw();
	void calculateAvgCpuTimestamps();
	void calculateAvgGpuTimestamps();
};