#pragma once

#include "../../base/DescriptorSet.hpp"
#include "../../base/UniformBuffer.hpp"
#include "../../../PipelineCreation.hpp"

class BufferDescriptorSet : public DescriptorSet {
public:
	BufferDescriptorSet(
		VulkanWindow* window, 
		VkDescriptorSetLayout* descSetLayout, 
		std::vector<DescriptorBufferSetting> descBufferSettings);

	void recreate();
private:
	std::vector<DescriptorBufferSetting> descBufferSettings;
};