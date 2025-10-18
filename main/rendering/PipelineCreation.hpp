#pragma once

#include "../vulkan/VulkanContext.hpp"
#include "../vulkan/objects/VkObjects.hpp"
#include "../vulkan/objects/VkImage.hpp"

class TextureBuffer;
class IBuffer;

struct DescriptorSetting {
	VkDescriptorType descriptorType;
	VkShaderStageFlags shaderStageFlags;
};

struct DescriptorImageSetting {
	TextureBuffer* textureBuffer;
	VkDescriptorType descriptorType;
	VkImageLayout imageLayout;
	VkSampler sampler;
};

struct DescriptorBufferSetting {
	IBuffer* buffer;
	VkDescriptorType descriptorType;
	VkDeviceSize range = VK_WHOLE_SIZE;
};

struct TextureBufferSetting {
	VkImageCreateFlags imageCreateFlags;
	VkFormat imageFormat;
	VkExtent2D imageExtent;
	std::uint32_t imageArrayLayers = 1;
	VkImageUsageFlags imageUsage;
	VkImageViewType viewType = VK_IMAGE_VIEW_TYPE_2D;
	VkImageAspectFlags viewAspectFlags;
	std::uint32_t subresourceLayerCount = 1;
	VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
	VkMemoryPropertyFlags allocationRequiredFlags = 0;
	VkMemoryPropertyFlags allocationPreferredFlags = 0;
	bool ignoreMipLevels = true;
};

vk::ShaderModule loadShaderModule(const VulkanDevice& device, const char* spirvPath);
vk::DescriptorSetLayout createDescriptorLayout(const VulkanDevice& device, std::vector<DescriptorSetting>& descriptorSettings);
vk::PipelineLayout createPipelineLayout(const VulkanDevice& device, std::vector<VkDescriptorSetLayout>& layouts, std::vector<VkPushConstantRange>& pushConstantRanges);
std::pair<vk::Image, vk::ImageView> createTextureBuffer(const VulkanContext& context, TextureBufferSetting textureSetting);

std::uint32_t computeMipLevels(std::uint32_t width, std::uint32_t height);

void createFramebuffers(
	const VulkanWindow& window, 
	std::vector<vk::Framebuffer>& framebuffers, 
	VkRenderPass renderPass, 
	std::vector<VkImageView>& imageViews, 
	VkExtent2D extent);

VkDescriptorSet createImageDescriptor(const VulkanWindow& window, VkDescriptorSetLayout descSetLayout, std::vector<DescriptorImageSetting>& imageViews);
void updateImageDescriptorSet(const VulkanDevice& device, VkDescriptorSet descriptorSet, std::vector<DescriptorImageSetting>& imageViews);
std::vector<VkDescriptorSet> createBufferDescriptors(const VulkanWindow& window, VkDescriptorSetLayout descSetLayout, std::vector<DescriptorBufferSetting>& buffers);
void updateBufferDescriptorSet(const VulkanDevice& device, VkDescriptorSet descriptorSet, std::vector<DescriptorBufferSetting>& buffers, int frame);