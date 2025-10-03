#pragma once

#include <memory>

#include "VulkanWindow.hpp"
#include "VulkanAllocator.hpp"

// TODO: seems like an extra layer of uselessness, just stick both in Renderer
// thats where they get initialised anyway
struct VulkanContext {
	std::unique_ptr<VulkanWindow> window;
	std::unique_ptr<VulkanAllocator> allocator;
};