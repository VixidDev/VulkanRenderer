#pragma once

#include "interfaces/ITextureBufferListener.hpp"
#include "../../../vulkan/objects/VkObjects.hpp"

class VulkanWindow;

class DescriptorSet {
public:
	DescriptorSet() = default;
	DescriptorSet(VulkanWindow* window, VkDescriptorSetLayout* descSetLayout);

	virtual ~DescriptorSet() = default;

	virtual void recreate();

	virtual VkDescriptorSet& getHandle(std::uint32_t frameIndex = 0);
protected:
	VulkanWindow* window = nullptr;
	VkDescriptorSetLayout* descSetLayout = nullptr;
};