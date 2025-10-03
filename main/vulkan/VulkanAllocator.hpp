#pragma once

#include <utility>

#include <volk/volk.h>
#include <vk_mem_alloc.h>

class VulkanWindow;

class VulkanAllocator {
public:
	VulkanAllocator() noexcept = default;
	VulkanAllocator(VulkanWindow* window);
	~VulkanAllocator();

	explicit VulkanAllocator(VmaAllocator) noexcept;

	// Move only
	VulkanAllocator(VulkanAllocator const&) = delete;
	VulkanAllocator& operator= (VulkanAllocator const&) = delete;
	VulkanAllocator(VulkanAllocator&&) noexcept;
	VulkanAllocator& operator = (VulkanAllocator&&) noexcept;

public:
	VmaAllocator allocator = VK_NULL_HANDLE;
};