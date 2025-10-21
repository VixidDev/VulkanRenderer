#pragma once

#include <map>
#include <string>

#include "../../base/PipelineLayout.hpp"

class VarianceShadowPipelineLayout : public PipelineLayout {
public:
	VarianceShadowPipelineLayout(VulkanWindow* window, std::map<std::string, vk::DescriptorSetLayout>* descriptorLayouts);

	void recreate();
private:
	std::map<std::string, vk::DescriptorSetLayout>* descriptorLayouts = nullptr;
};