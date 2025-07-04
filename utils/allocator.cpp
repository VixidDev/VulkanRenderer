#include "allocator.hpp"

// SOLUTION_TAGS: vulkan-(ex-[^123]|cw-.)

#include <utility>

#include <cassert>

#include "error.hpp"
#include "to_string.hpp"

namespace labutils
{
	Allocator::Allocator() noexcept = default;

	Allocator::~Allocator()
	{
		if( VK_NULL_HANDLE != allocator )
		{
			vmaDestroyAllocator( allocator );
		}
	}

	Allocator::Allocator( VmaAllocator aAllocator ) noexcept
		: allocator( aAllocator )
	{}

	Allocator::Allocator( Allocator&& aOther ) noexcept
		: allocator( std::exchange( aOther.allocator, VK_NULL_HANDLE ) )
	{}
	Allocator& Allocator::operator=( Allocator&& aOther ) noexcept
	{
		std::swap( allocator, aOther.allocator );
		return *this;
	}
}

namespace labutils
{
	Allocator create_allocator( VulkanContext const& aContext )
	{
		VkPhysicalDeviceProperties props{};
		vkGetPhysicalDeviceProperties( aContext.physicalDevice, &props );

		VmaVulkanFunctions functions{};
		functions.vkGetInstanceProcAddr   = vkGetInstanceProcAddr;
		functions.vkGetDeviceProcAddr     = vkGetDeviceProcAddr;

		VmaAllocatorCreateInfo allocInfo{};
		allocInfo.vulkanApiVersion  = props.apiVersion;
		allocInfo.physicalDevice    = aContext.physicalDevice;
		allocInfo.device            = aContext.device;
		allocInfo.instance          = aContext.instance;
		allocInfo.pVulkanFunctions  = &functions;

		std::printf("major: %d - minor: %d\n", VK_VERSION_MAJOR(allocInfo.vulkanApiVersion), VK_VERSION_MINOR(allocInfo.vulkanApiVersion));
		
		VmaAllocator allocator = VK_NULL_HANDLE;
		if( auto const res = vmaCreateAllocator( &allocInfo, &allocator ); VK_SUCCESS != res )
		{
			throw Error( "Unable to create allocator\n"
				"vmaCreateAllocator() returned %s", to_string(res).c_str()
			);
		}

		return Allocator( allocator );
	}
}

