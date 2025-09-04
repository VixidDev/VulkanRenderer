#pragma once

#include <utility>
#include <cassert>

#include <volk/volk.h>
#include <vk_mem_alloc.h>

#include "VulkanWindow.hpp"

class VulkanAllocator {
public:
	VulkanAllocator() noexcept, ~VulkanAllocator();

	explicit VulkanAllocator(VmaAllocator) noexcept;

	VulkanAllocator(VulkanAllocator const&) = delete;
	VulkanAllocator& operator= (VulkanAllocator const&) = delete;

	VulkanAllocator(VulkanAllocator&&) noexcept;
	VulkanAllocator& operator = (VulkanAllocator&&) noexcept;

public:
	VmaAllocator allocator = VK_NULL_HANDLE;
};

std::unique_ptr<VulkanAllocator> initialiseVulkanAllocator(const VulkanWindow& window);