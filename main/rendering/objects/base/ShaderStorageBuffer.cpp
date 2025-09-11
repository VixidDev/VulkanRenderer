#include "ShaderStorageBuffer.hpp"

ShaderStorageBuffer::ShaderStorageBuffer(VulkanContext* context) : context(context) {}

void ShaderStorageBuffer::update() {}

std::uint32_t ShaderStorageBuffer::getBufferSize() {
	return this->bufferSize;
}

VkBuffer ShaderStorageBuffer::getHandle() {
	return this->gpuBuffer.buffer;
}