#pragma once

#include <map>
#include <string>

#include "../../base/PipelineLayout.hpp"

struct debugStatePC {
	int lightCount;
	int debugState;
};

class DebugViewsPipelineLayout : public PipelineLayout {
public:
	DebugViewsPipelineLayout(VulkanWindow* window, std::map<std::string, vk::DescriptorSetLayout>* descriptorLayouts);

	void recreate();
private:
	std::map<std::string, vk::DescriptorSetLayout>* descriptorLayouts = nullptr;
};