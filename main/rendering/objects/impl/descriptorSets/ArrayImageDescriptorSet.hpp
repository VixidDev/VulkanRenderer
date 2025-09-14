#pragma once

#include "ImageDescriptorSet.hpp"

class ArrayImageDescriptorSet : public ImageDescriptorSet {
public:
	ArrayImageDescriptorSet(
		VulkanWindow* window,
		VkDescriptorSetLayout* descSetLayout,
		std::vector<DescriptorImageSetting> descImageSettings);

	void recreate() override;

	void derivedRecreate();

	std::vector<VkDescriptorSet>& getDescriptorSets();
private:
	std::vector<VkDescriptorSet> descriptorSets;
};