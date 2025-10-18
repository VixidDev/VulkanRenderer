#include "DescriptorSet.hpp"

DescriptorSet::DescriptorSet(VulkanWindow* window, VkDescriptorSetLayout* descSetLayout) 
	: window(window), descSetLayout(descSetLayout) {}

void DescriptorSet::recreate() {}

VkDescriptorSet& DescriptorSet::getHandle(std::uint32_t frameIndex) {
	static VkDescriptorSet dummy = VK_NULL_HANDLE;
	return dummy;
}
