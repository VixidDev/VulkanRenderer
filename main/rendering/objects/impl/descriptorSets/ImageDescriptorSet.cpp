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

	this->recreate();
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
	this->descriptorSet = createImageDescriptor(*this->window, *this->descSetLayout, this->descImageSettings);
}


