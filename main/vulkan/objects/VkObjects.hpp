#pragma once

#include <volk/volk.h>

#include <utility>

#include <cassert>

// Provide a small wrapper around Vulkan handles that destroys these when
// the object goes out of scope. 
//
// The wrappers are move-only, i.e., we cannot make copies of them. We are
// only allowed to pass ownership ("move") of the underlying handle to a 
// different object.
//
// The template takes three arguments:
//   - The type of Vulkan handle we want to wrap (VkRenderPass in the example)
//   - The type of the parent object (here VkDevice)
//   - The function that is used to destroy the Vulkan handle, which is 
//     vkDestroyRenderPass in this case. (Recall that this function takes
//     a VkDevice handle as its first argument.)

namespace vk {
	template<typename tParent, typename tHandle>
	using DestroyFn = void (*)(tParent, tHandle, const VkAllocationCallbacks*);

	template<typename tHandle, typename tParent, DestroyFn<tParent, tHandle>& tDestroyFn>
	class UniqueHandle final {
	public:
		UniqueHandle() noexcept = default;
		explicit UniqueHandle(tParent, tHandle = VK_NULL_HANDLE) noexcept;

		~UniqueHandle();

		UniqueHandle(UniqueHandle const&) = delete;
		UniqueHandle& operator= (UniqueHandle const&) = delete;

		UniqueHandle(UniqueHandle&&) noexcept;
		UniqueHandle& operator = (UniqueHandle&&) noexcept;

	public:
		tHandle handle = VK_NULL_HANDLE;

	private:
		tParent mParent = VK_NULL_HANDLE;
	};

	// Pre-defined wrapper types
	using RenderPass = UniqueHandle<VkRenderPass, VkDevice, vkDestroyRenderPass>;
	using Framebuffer = UniqueHandle<VkFramebuffer, VkDevice, vkDestroyFramebuffer>;

	using DescriptorPool = UniqueHandle<VkDescriptorPool, VkDevice, vkDestroyDescriptorPool>;
	using DescriptorSetLayout = UniqueHandle<VkDescriptorSetLayout, VkDevice, vkDestroyDescriptorSetLayout>;

	using Pipeline = UniqueHandle<VkPipeline, VkDevice, vkDestroyPipeline>;
	using PipelineLayout = UniqueHandle<VkPipelineLayout, VkDevice, vkDestroyPipelineLayout>;

	using ShaderModule = UniqueHandle<VkShaderModule, VkDevice, vkDestroyShaderModule>;
	using CommandPool = UniqueHandle<VkCommandPool, VkDevice, vkDestroyCommandPool>;

	using Fence = UniqueHandle<VkFence, VkDevice, vkDestroyFence>;
	using Semaphore = UniqueHandle<VkSemaphore, VkDevice, vkDestroySemaphore>;

	using ImageView = UniqueHandle<VkImageView, VkDevice, vkDestroyImageView>;
	using Sampler = UniqueHandle<VkSampler, VkDevice, vkDestroySampler>;

	using QueryPool = UniqueHandle<VkQueryPool, VkDevice, vkDestroyQueryPool>;
}

#include "VkObjects.inl"
