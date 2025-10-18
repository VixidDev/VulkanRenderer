#include "ImageDescriptorSet.hpp"

ImageDescriptorSet::ImageDescriptorSet(
	VulkanWindow* window,
	VkDescriptorSetLayout* descSetLayout,
	std::vector<DescriptorImageSetting> descImageSettings) : DescriptorSet(window, descSetLayout) 
{
	this->descImageSettings = descImageSettings;

	for (DescriptorImageSetting descImageSetting : this->descImageSettings) {
		descImageSetting.textureBuffer->addListener(this);
	}

	this->descriptorSet = createImageDescriptor(*this->window, *this->descSetLayout, this->descImageSettings);
}

ImageDescriptorSet::~ImageDescriptorSet() {
	for (DescriptorImageSetting descImageSetting : this->descImageSettings) {
		descImageSetting.textureBuffer->removeListener(this);
	}
}

void ImageDescriptorSet::onTextureBufferRecreated() {
	this->recreate();
}

void ImageDescriptorSet::recreate() {
	updateImageDescriptorSet(*this->window->getDevice(), this->descriptorSet, this->descImageSettings);
}

VkDescriptorSet& ImageDescriptorSet::getHandle(std::uint32_t frameIndex) {
	return this->descriptorSet;
}
