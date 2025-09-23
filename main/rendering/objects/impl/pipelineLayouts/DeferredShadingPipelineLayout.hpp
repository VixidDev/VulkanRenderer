#pragma once

#include <map>
#include <string>

#include "../../base/PipelineLayout.hpp"

class DeferredShadingPipelineLayout : public PipelineLayout {
public:
	DeferredShadingPipelineLayout(VulkanWindow* window,
		std::map<std::string, vk::DescriptorSetLayout>* descriptorLayouts,
		bool* shadowsEnabled);

	void recreate();
private:
	std::map<std::string, vk::DescriptorSetLayout>* descriptorLayouts = nullptr;
	bool* shadowsEnabled = nullptr;
};