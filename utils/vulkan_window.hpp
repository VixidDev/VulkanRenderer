#ifndef VULKAN_WINDOW_HPP_4A091E39_2253_474B_9E31_341B4E96E750
#define VULKAN_WINDOW_HPP_4A091E39_2253_474B_9E31_341B4E96E750
// SOLUTION_TAGS: vulkan-(ex-[^12]|cw-.)

#include <volk/volk.h>

#if !defined(GLFW_INCLUDE_NONE)
#	define GLFW_INCLUDE_NONE 1
#endif
#include <GLFW/glfw3.h>

#include <vector>
#include <cstdint>

#include "vulkan_context.hpp"

namespace labutils
{
	class VulkanWindow final : public VulkanContext
	{
		public:
			VulkanWindow(), ~VulkanWindow();

			// Move-only
			VulkanWindow( VulkanWindow const& ) = delete;
			VulkanWindow& operator= (VulkanWindow const&) = delete;

			VulkanWindow( VulkanWindow&& ) noexcept;
			VulkanWindow& operator= (VulkanWindow&&) noexcept;

		public:
			GLFWwindow* window = nullptr;
			VkSurfaceKHR surface = VK_NULL_HANDLE;

			std::uint32_t presentFamilyIndex = 0;
			VkQueue presentQueue = VK_NULL_HANDLE;

			VkSwapchainKHR swapchain = VK_NULL_HANDLE;
			std::vector<VkImage> swapImages;
			std::vector<VkImageView> swapViews;

			VkFormat swapchainFormat;
			VkExtent2D swapchainExtent;
	};

	VulkanWindow make_vulkan_window();


	struct SwapChanges
	{
		bool changedSize : 1;
		bool changedFormat: 1;
	};

	SwapChanges recreate_swapchain( VulkanWindow& );
}

#endif // VULKAN_WINDOW_HPP_4A091E39_2253_474B_9E31_341B4E96E750
