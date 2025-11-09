#include "BufferDescriptorSet.hpp"

BufferDescriptorSet::BufferDescriptorSet(
	VulkanWindow* window,
	VkDescriptorSetLayout* descSetLayout,
	std::vector<DescriptorBufferSetting> descBufferSettings
) : DescriptorSet(window, descSetLayout) 
{
	this->descBufferSettings = descBufferSettings;

	this->descriptorSets = createBufferDescriptors(*this->window, *this->descSetLayout, this->descBufferSettings);
}

void BufferDescriptorSet::recreate() {
	for (std::size_t i = 0; i < this->descriptorSets.size(); i++) {
		updateBufferDescriptorSet(*this->window->getDevice(), this->descriptorSets[i], this->descBufferSettings, (int)i);
	}
}

VkDescriptorSet& BufferDescriptorSet::getHandle(std::uint32_t frameIndex) {
	return this->descriptorSets[frameIndex];
}
