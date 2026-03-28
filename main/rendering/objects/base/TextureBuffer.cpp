#include "TextureBuffer.hpp"

TextureBuffer::TextureBuffer(VulkanContext* context) : context(context) {}

void TextureBuffer::recreate() {
	for (ITextureBufferListener* listener : this->listeners) {
		if (!listener) continue;
		listener->onTextureBufferRecreated();
	}
}

void TextureBuffer::addListener(ITextureBufferListener* listener) {
	this->listeners.push_back(listener);
}

void TextureBuffer::removeListener(ITextureBufferListener* listener) {
	this->listeners.erase(std::remove(this->listeners.begin(), this->listeners.end(), listener), this->listeners.end());
}

vk::Image& TextureBuffer::getImage() {
	return this->image;
}

vk::ImageView& TextureBuffer::getImageView() {
	return this->imageView;
}

TextureBuffer::Builder* TextureBuffer::Builder::withDescription(TextureDesc textureDesc) {
	this->textureDesc = textureDesc;
	return this;
}

TextureBuffer::Builder* TextureBuffer::Builder::withExtent(ExtentRatio extent) {
	this->extent = extent;
	return this;
}

TextureBuffer::Builder* TextureBuffer::Builder::hasFutureUse(TextureUseFlags futureUse) {
	this->futureUse = futureUse;
	return this;
}

TextureBuffer TextureBuffer::Builder::build() {

}