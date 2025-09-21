#pragma once

#include <map>
#include <string>

#include "../../base/PipelineLayout.hpp"

class ForwardPipelineLayout : public PipelineLayout {
public:
	ForwardPipelineLayout(VulkanWindow* window, 
		std::map<std::string, vk::DescriptorSetLayout>* descriptorLayouts, 
		bool* shadowsEnabled);

	void recreate();
private:
	std::map<std::string, vk::DescriptorSetLayout>* descriptorLayouts = nullptr;
	bool* shadowsEnabled = nullptr;
};