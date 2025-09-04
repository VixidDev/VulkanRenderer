#pragma once

#include "interfaces/ITextureBufferListener.hpp"
#include "../../../vulkan/objects/VkObjects.hpp"

class VulkanWindow;

class DescriptorSet {
public:
	DescriptorSet() = default;
	DescriptorSet(VulkanWindow* window, VkDescriptorSetLayout* descSetLayout);

	virtual ~DescriptorSet() = default;

	void recreate();

	VkDescriptorSet& getHandle();
protected:
	VulkanWindow* window = nullptr;
	VkDescriptorSetLayout* descSetLayout = nullptr;

	VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
};