#pragma once

#include <map>
#include <string>

#include "../../base/PipelineLayout.hpp"

class SkyboxPipelineLayout : public PipelineLayout {
public:
	SkyboxPipelineLayout(VulkanWindow* window,
		std::map<std::string, vk::DescriptorSetLayout>* descriptorLayouts);

	void recreate();
private:
	std::map<std::string, vk::DescriptorSetLayout>* descriptorLayouts = nullptr;
};