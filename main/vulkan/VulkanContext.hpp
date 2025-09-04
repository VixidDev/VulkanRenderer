#pragma once

#include <memory>

#include "VulkanWindow.hpp"
#include "VulkanAllocator.hpp"

struct VulkanContext {
	std::unique_ptr<VulkanWindow> window;
	std::unique_ptr<VulkanAllocator> allocator;
};