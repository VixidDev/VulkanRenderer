#include "ArrayImageDescriptorSet.hpp"

#include "../../base/ArrayTextureBuffer.hpp"

#include "../../../../vulkan/VulkanDevice.hpp"
#include "../../../../vulkan/VkUtils.hpp"

ArrayImageDescriptorSet::ArrayImageDescriptorSet(
	VulkanWindow* window,
	VkDescriptorSetLayout* descSetLayout,
	std::vector<DescriptorImageSetting> descImageSettings) : ImageDescriptorSet(window, descSetLayout, descImageSettings) 
{
	for (DescriptorImageSetting descImageSetting : this->descImageSettings) {
		ArrayTextureBuffer* arrayTextureBuffer = dynamic_cast<ArrayTextureBuffer*>(descImageSetting.textureBuffer);

		for (vk::ImageView& imageView : arrayTextureBuffer->getFramebufferViews()) {
			VkDescriptorSet imageDescriptor = VkUtils::createDescriptorSet(*this->window, this->window->getDevice()->getDescPool(), *this->descSetLayout);
			{
				updateDescriptorSet(descImageSetting, imageView, imageDescriptor);
			}

			this->descriptorSets.emplace_back(imageDescriptor);
		}
	}
}

void ArrayImageDescriptorSet::recreate() {
	for (DescriptorImageSetting descImageSetting : this->descImageSettings) {
		ArrayTextureBuffer* arrayTextureBuffer = dynamic_cast<ArrayTextureBuffer*>(descImageSetting.textureBuffer);

		for (int i = 0; i < arrayTextureBuffer->getFramebufferViews().size(); i++) {
			vk::ImageView& imageView = arrayTextureBuffer->getFramebufferViews().at(i);

			this->updateDescriptorSet(descImageSetting, imageView, this->descriptorSets.at(i));
		}
	}

	ImageDescriptorSet::recreate();
}

void ArrayImageDescriptorSet::updateDescriptorSet(DescriptorImageSetting descImageSetting, vk::ImageView& imageView, VkDescriptorSet descriptorSet) {
	VkDescriptorImageInfo descImageInfo = {
		.sampler = descImageSetting.sampler,
		.imageView = imageView.handle,
		.imageLayout = descImageSetting.imageLayout
	};

	VkWriteDescriptorSet desc = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = descriptorSet,
		.dstBinding = 0,
		.descriptorCount = 1,
		.descriptorType = descImageSetting.descriptorType,
		.pImageInfo = &descImageInfo
	};

	vkUpdateDescriptorSets(this->window->getDevice()->getDevice(), 1, &desc, 0, nullptr);
}

std::vector<VkDescriptorSet>& ArrayImageDescriptorSet::getDescriptorSets() {
	return this->descriptorSets;
}
