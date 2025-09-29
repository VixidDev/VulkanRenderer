#include "BufferDescriptorSet.hpp"

BufferDescriptorSet::BufferDescriptorSet(
	VulkanWindow* window,
	VkDescriptorSetLayout* descSetLayout,
	std::vector<DescriptorBufferSetting> descBufferSettings
) : DescriptorSet(window, descSetLayout) 
{
	this->descBufferSettings = descBufferSettings;

	this->recreate();
}

void BufferDescriptorSet::recreate() {
	this->descriptorSet = createBufferDescriptor(*this->window, *this->descSetLayout, this->descBufferSettings);
}
