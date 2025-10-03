#include "PipelineCreation.hpp"

#include <bit>

#include "Error.hpp"
#include "toString.hpp"
#include "../vulkan/VulkanWindow.hpp"
#include "../vulkan/VulkanDevice.hpp"
#include "../vulkan/Swapchain.hpp"
#include "objects/base/TextureBuffer.hpp"
#include "objects/base/UniformBuffer.hpp"

vk::ShaderModule loadShaderModule(const VulkanDevice& device, const char* spirvPath) {
	assert(spirvPath);

	if (std::FILE* fin = std::fopen(spirvPath, "rb")) {
		std::fseek(fin, 0, SEEK_END);
		const auto bytes = std::size_t(std::ftell(fin));
		std::fseek(fin, 0, SEEK_SET);

		assert(0 == bytes % 4);
		const auto words = bytes / 4;

		std::vector<std::uint32_t> code(words);

		std::size_t offset = 0;
		while (offset != words) {
			const auto read = std::fread(code.data() + offset, sizeof(std::uint32_t), words - offset, fin);

			if (0 == read) {
				const auto err = std::ferror(fin), eof = std::feof(fin);
				std::fclose(fin);

				throw Utils::Error("Utils::Error reading '%s': fUtils::Error: %d, feof = %d", spirvPath, err, eof);
			}

			offset += read;
		}

		std::fclose(fin);

		VkShaderModuleCreateInfo moduleInfo{};
		moduleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		moduleInfo.codeSize = bytes;
		moduleInfo.pCode = code.data();

		VkShaderModule smod = VK_NULL_HANDLE;
		if (const auto res = vkCreateShaderModule(device.getDevice(), &moduleInfo, nullptr, &smod); VK_SUCCESS != res) {
			throw Utils::Error("Unable to create shader module from %s\n vkShaderCreateShaderModule() returned %s", spirvPath, Utils::toString(res).c_str());
		}

		return vk::ShaderModule(device.getDevice(), smod);
	}

	throw Utils::Error("Cannot open '%s' for reading", spirvPath);
}

vk::DescriptorSetLayout createDescriptorLayout(const VulkanDevice& device, std::vector<DescriptorSetting>& descriptorSettings) {
	std::vector<VkDescriptorSetLayoutBinding> layoutBindings;

	for (std::size_t i = 0; i < descriptorSettings.size(); i++) {
		VkDescriptorSetLayoutBinding binding{};
		binding.binding = static_cast<std::uint32_t>(i);
		binding.descriptorType = descriptorSettings[i].descriptorType;
		binding.descriptorCount = 1;
		binding.stageFlags = descriptorSettings[i].shaderStageFlags;

		layoutBindings.emplace_back(binding);
	}

	VkDescriptorSetLayoutCreateInfo layoutInfo{};
	layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layoutInfo.bindingCount = static_cast<std::uint32_t>(layoutBindings.size());
	layoutInfo.pBindings = layoutBindings.data();

	VkDescriptorSetLayout layout = VK_NULL_HANDLE;
	if (const auto res = vkCreateDescriptorSetLayout(device.getDevice(), &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
		throw Utils::Error("Unable to create descriptor set layout\n vkCreateDescriptorSetLayout() returned %s", Utils::toString(res).c_str());

	return vk::DescriptorSetLayout(device.getDevice(), layout);
}

vk::PipelineLayout createPipelineLayout(const VulkanDevice& device, std::vector<VkDescriptorSetLayout>& aDescriptorSetLayouts, std::vector<VkPushConstantRange>& aPushConstantRanges) {
	VkPipelineLayoutCreateInfo layoutInfo{};
	layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	layoutInfo.setLayoutCount = static_cast<std::uint32_t>(aDescriptorSetLayouts.size());
	layoutInfo.pSetLayouts = aDescriptorSetLayouts.data();
	layoutInfo.pushConstantRangeCount = static_cast<std::uint32_t>(aPushConstantRanges.size());
	layoutInfo.pPushConstantRanges = aPushConstantRanges.data();

	VkPipelineLayout layout = VK_NULL_HANDLE;
	if (const auto res = vkCreatePipelineLayout(device.getDevice(), &layoutInfo, nullptr, &layout); VK_SUCCESS != res) {
		throw Utils::Error("Unable to create pipeline layout\n vkCreatePipelineLayout() returned %s", Utils::toString(res).c_str());
	}

	return vk::PipelineLayout(device.getDevice(), layout);
}

// Should only be used for render pass attachments
std::pair<vk::Image, vk::ImageView> createTextureBuffer(const VulkanContext& context, TextureBufferSetting aBufferSetting) {
	std::uint32_t mipLevels = 1;

	if (!aBufferSetting.ignoreMipLevels)
		mipLevels = computeMipLevels(aBufferSetting.imageExtent.width, aBufferSetting.imageExtent.height);

	VkImageCreateInfo imageInfo{};
	imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageInfo.flags = aBufferSetting.imageCreateFlags;
	imageInfo.imageType = VK_IMAGE_TYPE_2D;
	imageInfo.format = aBufferSetting.imageFormat;
	imageInfo.extent.width = aBufferSetting.imageExtent.width;
	imageInfo.extent.height = aBufferSetting.imageExtent.height;
	imageInfo.extent.depth = 1;
	imageInfo.mipLevels = mipLevels;
	imageInfo.arrayLayers = aBufferSetting.imageArrayLayers;
	imageInfo.samples = aBufferSetting.samples;
	imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageInfo.usage = aBufferSetting.imageUsage;
	imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

	VmaAllocationCreateInfo allocInfo{};
	allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
	allocInfo.requiredFlags = aBufferSetting.allocationRequiredFlags;
	allocInfo.preferredFlags = aBufferSetting.allocationPreferredFlags;

	VkImage image = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;

	if (const auto res = vmaCreateImage(context.allocator->allocator, &imageInfo, &allocInfo, &image, &allocation, nullptr); VK_SUCCESS != res)
		throw Utils::Error("Unable to allocate depth buffer image.\n vmaCreateImage() returned %s\n", Utils::toString(res).c_str());

	vk::Image Image(context.allocator->allocator, image, allocation);

	VkImageViewCreateInfo viewInfo{};
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.image = Image.image;
	viewInfo.viewType = aBufferSetting.viewType;
	viewInfo.format = aBufferSetting.imageFormat;
	viewInfo.components = VkComponentMapping{};
	viewInfo.subresourceRange = VkImageSubresourceRange{ aBufferSetting.viewAspectFlags, 0, 1, 0, aBufferSetting.subresourceLayerCount };

	VkImageView view = VK_NULL_HANDLE;
	if (const auto res = vkCreateImageView(context.window->getDevice()->getDevice(), &viewInfo, nullptr, &view); VK_SUCCESS != res)
		throw Utils::Error("Unable to create image view.\n vkCreateImageView() returned %s", Utils::toString(res).c_str());

	return { std::move(Image), vk::ImageView(context.window->getDevice()->getDevice(), view) };
}

std::uint32_t computeMipLevels(std::uint32_t width, std::uint32_t height) {
	const std::uint32_t bits = width | height;
	const std::uint32_t leadingZeros = std::countl_zero(bits);
	return 32 - leadingZeros;
}

void createFramebuffers(
	const VulkanWindow& window, 
	std::vector<vk::Framebuffer>& framebuffers, 
	VkRenderPass renderPass, 
	std::vector<VkImageView>& imageViews, 
	VkExtent2D extent) 
{
	assert(framebuffers.empty());

	for (std::size_t i = 0; i < window.getSwapchain()->getViews().size(); ++i) {
		VkFramebufferCreateInfo fbInfo{};
		fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		fbInfo.flags = 0;
		fbInfo.renderPass = renderPass;
		fbInfo.attachmentCount = static_cast<std::uint32_t>(imageViews.size());
		fbInfo.pAttachments = imageViews.data();
		fbInfo.width = extent.width;
		fbInfo.height = extent.height;
		fbInfo.layers = 1;

		VkFramebuffer fb = VK_NULL_HANDLE;
		if (const auto res = vkCreateFramebuffer(window.getDevice()->getDevice(), &fbInfo, nullptr, &fb); VK_SUCCESS != res)
			throw Utils::Error("Unable to create framebuffer for swap chain image %zu\n vkCreateFramebuffer() returned %s", i, Utils::toString(res).c_str());

		framebuffers.emplace_back(vk::Framebuffer(window.getDevice()->getDevice(), fb));
	}

	assert(window.getSwapchain()->getViews().size() == framebuffers.size());
}

VkDescriptorSet createImageDescriptor(const VulkanWindow& window, VkDescriptorSetLayout descSetLayout, std::vector<DescriptorImageSetting>& imageViews) {
	VkDescriptorSet imageDescriptor = VkUtils::createDescriptorSet(window, window.getDevice()->getDescPool(), descSetLayout);
	{
		std::vector<VkDescriptorImageInfo> descImageInfos;
		std::vector<VkWriteDescriptorSet> descs;

		for (std::size_t i = 0; i < imageViews.size(); i++) {
			VkDescriptorImageInfo descImageInfo{};
			descImageInfo.imageLayout = imageViews[i].imageLayout;
			descImageInfo.imageView = imageViews[i].textureBuffer->getImageView().handle;
			descImageInfo.sampler = imageViews[i].sampler;
			descImageInfos.push_back(descImageInfo);
		}

		for (std::size_t i = 0; i < imageViews.size(); i++) {
			VkWriteDescriptorSet desc{};
			desc.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			desc.dstSet = imageDescriptor;
			desc.dstBinding = i;
			desc.descriptorType = imageViews[i].descriptorType;
			desc.descriptorCount = 1;
			desc.pImageInfo = &descImageInfos[i];
			descs.push_back(desc);
		}

		std::size_t numSets = descs.size();
		vkUpdateDescriptorSets(window.getDevice()->getDevice(), numSets, descs.data(), 0, nullptr);
	}

	return imageDescriptor;
}

VkDescriptorSet createBufferDescriptor(const VulkanWindow& window, VkDescriptorSetLayout descSetLayout, std::vector<DescriptorBufferSetting>& buffers) {
	VkDescriptorSet bufferDescriptor = VkUtils::createDescriptorSet(window, window.getDevice()->getDescPool(), descSetLayout);
	{
		std::vector<VkDescriptorBufferInfo> descBufferInfos;
		std::vector<VkWriteDescriptorSet> descs;

		for (std::size_t i = 0; i < buffers.size(); i++) {
			bool isStorageBuffer = buffers[i].descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			bool needsVkWholeSize = isStorageBuffer && buffers[i].bufferHandle == VK_NULL_HANDLE;

			VkDescriptorBufferInfo descBufferInfo{};
			descBufferInfo.buffer = buffers[i].bufferHandle;
			descBufferInfo.offset = 0;
			descBufferInfo.range = needsVkWholeSize ? VK_WHOLE_SIZE : buffers[i].range;
			descBufferInfos.push_back(descBufferInfo);
		}

		for (std::size_t i = 0; i < buffers.size(); i++) {
			VkWriteDescriptorSet desc{};
			desc.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			desc.dstSet = bufferDescriptor;
			desc.dstBinding = i;
			desc.descriptorType = buffers[i].descriptorType;
			desc.descriptorCount = 1;
			desc.pBufferInfo = &descBufferInfos[i];
			descs.push_back(desc);
		}

		std::size_t numSets = descs.size();
		vkUpdateDescriptorSets(window.getDevice()->getDevice(), numSets, descs.data(), 0, nullptr);
	}

	return bufferDescriptor;
}