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

	void recreate() override;
	VkDescriptorSet& getHandle(std::uint32_t frameIndex) override;
private:
	std::vector<DescriptorBufferSetting> descBufferSettings;

	// A VkDescriptorSet per frame in flight
	std::vector<VkDescriptorSet> descriptorSets;
};