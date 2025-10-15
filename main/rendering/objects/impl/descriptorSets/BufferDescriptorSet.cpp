#include "BufferDescriptorSet.hpp"

BufferDescriptorSet::BufferDescriptorSet(
	VulkanWindow* window,
	VkDescriptorSetLayout* descSetLayout,
	std::vector<DescriptorBufferSetting> descBufferSettings
) : DescriptorSet(window, descSetLayout) 
{
	this->descBufferSettings = descBufferSettings;

	this->descriptorSet = createBufferDescriptor(*this->window, *this->descSetLayout, this->descBufferSettings);
}

void BufferDescriptorSet::recreate() {
	updateBufferDescriptorSet(*this->window->getDevice(), this->descriptorSet, this->descBufferSettings);
}
