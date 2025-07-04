#ifndef ALLOCATOR_HPP_9E06592D_0990_41CD_AA6E_73AF54B53994
#define ALLOCATOR_HPP_9E06592D_0990_41CD_AA6E_73AF54B53994
// SOLUTION_TAGS: vulkan-(ex-[^123]|cw-.)

#include <volk/volk.h>
#include <vk_mem_alloc.h>

#include <utility>

#include <cassert>

#include "vulkan_context.hpp"

namespace labutils
{
	class Allocator
	{
		public:
			Allocator() noexcept, ~Allocator();

			explicit Allocator( VmaAllocator ) noexcept;

			Allocator( Allocator const& ) = delete;
			Allocator& operator= (Allocator const&) = delete;

			Allocator( Allocator&& ) noexcept;
			Allocator& operator = (Allocator&&) noexcept;

		public:
			VmaAllocator allocator = VK_NULL_HANDLE;
	};

	Allocator create_allocator( VulkanContext const& );
}

#endif // ALLOCATOR_HPP_9E06592D_0990_41CD_AA6E_73AF54B53994
