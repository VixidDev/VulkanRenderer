#pragma once

class IBuffer {
public:
	virtual ~IBuffer() = default;
	virtual VkBuffer getHandle(std::uint32_t frameIndex) const = 0;
};