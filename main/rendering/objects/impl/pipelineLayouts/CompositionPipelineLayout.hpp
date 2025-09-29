#pragma once

#include <map>
#include <string>

#include "../../base/PipelineLayout.hpp"

// I should remove this (and probably other layouts)
// that are 'static' i.e. aren't affected by a change in swapchain and can stay the same
// through the entire application's lifetime, that way no need to create all these classes
// and files for minor differentiating layouts
class CompositionPipelineLayout : public PipelineLayout {
public:
	CompositionPipelineLayout(VulkanWindow* window,
		std::map<std::string, vk::DescriptorSetLayout>* descriptorLayouts);

	void recreate();
private:
	std::map<std::string, vk::DescriptorSetLayout>* descriptorLayouts = nullptr;
};