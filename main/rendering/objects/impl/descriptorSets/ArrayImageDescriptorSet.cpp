#include "ArrayImageDescriptorSet.hpp"

#include "../../base/ArrayTextureBuffer.hpp"

#include "../../../../vulkan/VulkanDevice.hpp"
#include "../../../../vulkan/VkUtils.hpp"

ArrayImageDescriptorSet::ArrayImageDescriptorSet(
	VulkanWindow* window,
	VkDescriptorSetLayout* descSetLayout,
	std::vector<DescriptorImageSetting> descImageSettings) : ImageDescriptorSet(window, descSetLayout, descImageSettings) 
{
	this->derivedRecreate();
}

void ArrayImageDescriptorSet::derivedRecreate() {
	this->descriptorSets.clear();

	for (DescriptorImageSetting descImageSetting : this->descImageSettings) {
		ArrayTextureBuffer* arrayTextureBuffer = dynamic_cast<ArrayTextureBuffer*>(descImageSetting.textureBuffer);

		for (vk::ImageView& imageView : arrayTextureBuffer->getFramebufferViews()) {
			VkDescriptorSet imageDescriptor = VkUtils::createDescriptorSet(*this->window, this->window->device->descPool, *this->descSetLayout);
			{
				VkDescriptorImageInfo descImageInfo = {
					.sampler = descImageSetting.sampler,
					.imageView = imageView.handle,
					.imageLayout = descImageSetting.imageLayout
				};

				VkWriteDescriptorSet desc = {
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = imageDescriptor,
					.dstBinding = 0,
					.descriptorCount = 1,
					.descriptorType = descImageSetting.descriptorType,
					.pImageInfo = &descImageInfo
				};

				vkUpdateDescriptorSets(this->window->device->device, 1, &desc, 0, nullptr);
			}

			this->descriptorSets.emplace_back(imageDescriptor);
		}
	}
}

void ArrayImageDescriptorSet::recreate() {
	this->derivedRecreate();
	ImageDescriptorSet::recreate();
}

std::vector<VkDescriptorSet>& ArrayImageDescriptorSet::getDescriptorSets() {
	return this->descriptorSets;
}
