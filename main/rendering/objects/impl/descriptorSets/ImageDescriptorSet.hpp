#pragma once

#include "../../base/DescriptorSet.hpp"
#include "../../base/interfaces/ITextureBufferListener.hpp"
#include "../../base/TextureBuffer.hpp"
#include "../../../PipelineCreation.hpp"

class ImageDescriptorSet : public DescriptorSet, public ITextureBufferListener {
public:
	ImageDescriptorSet(
		VulkanWindow* window,
		VkDescriptorSetLayout* descSetLayout,
		std::vector<DescriptorImageSetting> descImageSettings);

	~ImageDescriptorSet();

	void onTextureBufferRecreated() override;

	void recreate();
private:
	std::vector<DescriptorImageSetting> descImageSettings;
};