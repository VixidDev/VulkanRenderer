#include "DescriptorSet.hpp"

DescriptorSet::DescriptorSet(VulkanWindow* window, VkDescriptorSetLayout* descSetLayout) 
	: window(window), descSetLayout(descSetLayout) {}

void DescriptorSet::recreate() {}

VkDescriptorSet& DescriptorSet::getHandle() {
	return this->descriptorSet;
}
