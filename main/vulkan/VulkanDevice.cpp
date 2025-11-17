#include "VulkanDevice.hpp"

#include "VulkanWindow.hpp"
#include "ContextHelpers.hpp"

#include "Error.hpp"
#include "toString.hpp"

VulkanDevice::VulkanDevice(VkDevice device, const VulkanWindow& window) noexcept : device(device) {}

VulkanDevice::VulkanDevice(VulkanDevice&& other) noexcept
	: physicalDevice(std::exchange(other.physicalDevice, VK_NULL_HANDLE))
	, device(std::exchange(other.device, VK_NULL_HANDLE))
	, graphicsQueue(std::exchange(other.graphicsQueue, VK_NULL_HANDLE))
	, presentQueue(std::exchange(other.presentQueue, VK_NULL_HANDLE))
	, cmdPool(std::exchange(other.cmdPool, VK_NULL_HANDLE))
	, descPool(std::exchange(other.descPool, VK_NULL_HANDLE)) {}

VulkanDevice& VulkanDevice::operator=(VulkanDevice&& other) noexcept {
	std::swap(physicalDevice, other.physicalDevice);
	std::swap(device, other.device);
	std::swap(graphicsQueue, other.graphicsQueue);
	std::swap(presentQueue, other.presentQueue);
	std::swap(cmdPool, other.cmdPool);
	std::swap(descPool, other.descPool);
	return *this;
}

VulkanDevice::VulkanDevice(VkInstance instance, VkSurfaceKHR surface) {
	// Select appropriate Vulkan device
	this->selectPhysicalDevice(instance, surface);
	if (this->physicalDevice == VK_NULL_HANDLE)
		throw Utils::Error("No suitable physical device found!");

	vkGetPhysicalDeviceProperties(this->physicalDevice, &this->deviceProperties);
	std::fprintf(stderr, "Selected device: %s (%d.%d.%d)\n",
		this->deviceProperties.deviceName,
		VK_API_VERSION_MAJOR(this->deviceProperties.apiVersion),
		VK_API_VERSION_MINOR(this->deviceProperties.apiVersion),
		VK_API_VERSION_PATCH(this->deviceProperties.apiVersion));

	// Create a logical device
	// Enable required extensions. The device selection method ensures that
	// the VK_KHR_swapchain extension is present, so we can safely just
	// request it without further checks.
	this->enabledDevExtensions.emplace_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

	for (const auto& ext : this->enabledDevExtensions)
		std::fprintf(stderr, "Enabling device extension: %s\n", ext);

	// We need one or two queues:
	// - best case: one GRAPHICS queue that can present
	// - otherwise: one GRAPHICS queue and any queue that can present
	if (const std::optional<std::uint32_t> index = this->findQueueFamily(this->physicalDevice, VK_QUEUE_GRAPHICS_BIT, surface)) {
		this->graphicsFamilyIndex = *index;
		this->queueFamilyIndices.emplace_back(*index);
	} else {
		std::optional<std::uint32_t> graphics = this->findQueueFamily(this->physicalDevice, VK_QUEUE_GRAPHICS_BIT);
		std::optional<std::uint32_t> present = this->findQueueFamily(this->physicalDevice, 0, surface);

		assert(graphics && present);

		this->graphicsFamilyIndex = *graphics;
		this->presentFamilyIndex = *present;
		this->queueFamilyIndices.emplace_back(*graphics);
		this->queueFamilyIndices.emplace_back(*present);
	}

	this->createLogicalDevice();

	// Retrieve VkQueues
	vkGetDeviceQueue(this->device, this->graphicsFamilyIndex, 0, &this->graphicsQueue);

	assert(this->graphicsQueue != VK_NULL_HANDLE && "Graphics queue needs to exist!");

	if (this->queueFamilyIndices.size() >= 2)
		vkGetDeviceQueue(this->device, this->presentFamilyIndex, 0, &this->presentQueue);
	else {
		this->presentFamilyIndex = this->graphicsFamilyIndex;
		this->presentQueue = this->graphicsQueue;
	}

	// Create pools
	this->createCommandPool();
	this->createDescriptorPool();
}

VulkanDevice::~VulkanDevice() {
	if (this->descPool != VK_NULL_HANDLE)
		vkDestroyDescriptorPool(this->device, this->descPool, nullptr);

	if (this->cmdPool != VK_NULL_HANDLE)
		vkDestroyCommandPool(this->device, this->cmdPool, nullptr);

	if (this->device != VK_NULL_HANDLE)
		vkDestroyDevice(this->device, nullptr);
}

void VulkanDevice::selectPhysicalDevice(VkInstance instance, VkSurfaceKHR surface) {
	std::uint32_t numDevices = 0;
	if (const VkResult res = vkEnumeratePhysicalDevices(instance, &numDevices, nullptr); VK_SUCCESS != res)
		throw Utils::Error("Unable to get physical device count\nvkEnumeratePhysicalDevices() returned %s", Utils::toString(res).c_str());

	std::vector<VkPhysicalDevice> devices(numDevices, VK_NULL_HANDLE);
	if (const VkResult res = vkEnumeratePhysicalDevices(instance, &numDevices, devices.data()); VK_SUCCESS != res)
		throw Utils::Error("Unable to get physical device list\nvkEnumeratePhysicalDevices() returned %s", Utils::toString(res).c_str());

	float bestScore = -1.f;
	VkPhysicalDevice bestDevice = VK_NULL_HANDLE;

	for (const VkPhysicalDevice device : devices) {
		const float score = this->scoreDevice(device, surface);
		if (score > bestScore) {
			bestScore = score;
			bestDevice = device;
		}
	}

	this->physicalDevice = bestDevice;
}

float VulkanDevice::scoreDevice(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) {
	VkPhysicalDeviceProperties props;
	vkGetPhysicalDeviceProperties(physicalDevice, &props);

	// Only consider Vulkan 1.1 devices
	const auto major = VK_API_VERSION_MAJOR(props.apiVersion);
	const auto minor = VK_API_VERSION_MINOR(props.apiVersion);

	if (major < 1 || (major == 1 && minor < 2)) {
		std::fprintf(stderr, "Info: Discarding device '%s': insufficient vulkan version\n", props.deviceName);
		return -1.f;
	}

	const std::unordered_set<std::string> supportedExtensions = Utils::getDeviceExtensions(physicalDevice);

	if (!supportedExtensions.count(VK_KHR_SWAPCHAIN_EXTENSION_NAME)) {
		std::fprintf(stderr, "Info: Discarding device '%s': extension %s missing\n", props.deviceName, VK_KHR_SWAPCHAIN_EXTENSION_NAME);
		return -1.0f;
	}

	if (!this->findQueueFamily(physicalDevice, 0, surface)) {
		std::fprintf(stderr, "Info: Discarding device '%s': can't present to surface\n", props.deviceName);
		return -1.0f;
	}

	if (!this->findQueueFamily(physicalDevice, VK_QUEUE_GRAPHICS_BIT)) {
		std::fprintf(stderr, "Info: Discarding device '%s': no graphics queue family\n", props.deviceName);
		return -1.0f;
	}

	// Discrete GPU > Integrated GPU > others
	float score = 0.f;

	if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
		score += 500.f;
	else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
		score += 100.f;

	return score;
}

// Note: this finds *any* queue that supports the queueFlags. As such,
// findQueueFamily( ..., VK_QUEUE_TRANSFER_BIT, ... );
// might return a GRAPHICS queue family, since GRAPHICS queues typically
// also set TRANSFER (and indeed most other operations; GRAPHICS queues are
// required to support those operations regardless). If you wanted to find
// a dedicated TRANSFER queue (e.g., such as those that exist on NVIDIA
// GPUs), you would need to use different logic.
std::optional<std::uint32_t> VulkanDevice::findQueueFamily(VkPhysicalDevice physicalDevice, VkQueueFlags queueFlags, VkSurfaceKHR surface) {
	std::uint32_t numQueues = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &numQueues, nullptr);

	std::vector<VkQueueFamilyProperties> families(numQueues);
	vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &numQueues, families.data());

	for (std::uint32_t i = 0; i < numQueues; ++i) {
		const VkQueueFamilyProperties& family = families[i];

		if (queueFlags == (queueFlags & family.queueFlags)) {
			if (surface == VK_NULL_HANDLE)
				return i;

			VkBool32 supported = VK_FALSE;
			const VkResult res = vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, surface, &supported);

			if (VK_SUCCESS == res && supported)
				return i;
		}
	}

	return {};
}

void VulkanDevice::createLogicalDevice() {
	if (this->queueFamilyIndices.empty())
		throw Utils::Error("createLogicalDevice(): no queues requested");

	float queuePriorities[1] = { 1.f };

	std::vector<VkDeviceQueueCreateInfo> queueInfos(this->queueFamilyIndices.size());
	for (std::size_t i = 0; i < this->queueFamilyIndices.size(); ++i) {
		VkDeviceQueueCreateInfo& queueInfo = queueInfos[i];
		queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queueInfo.queueFamilyIndex = this->queueFamilyIndices[i];
		queueInfo.queueCount = 1;
		queueInfo.pQueuePriorities = queuePriorities;
	}

	VkPhysicalDeviceRobustness2FeaturesEXT robustness2Features = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT
	};
	VkPhysicalDeviceFeatures2 deviceFeatures = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
		.pNext = &robustness2Features
	};
	// Fills deviceFeatures (and any of its pNext's) with all of the GPU's possible features,
	// we don't want to have every possible feature so we will check individual features and
	// enable them in our own VkPhysicalDeviceFeatures2 struct
	vkGetPhysicalDeviceFeatures2(this->physicalDevice, &deviceFeatures);

	VkPhysicalDeviceRobustness2FeaturesEXT enabledRobustFeatures = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT
	};
	VkPhysicalDeviceFeatures2 enabledFeatures = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
		.pNext = &enabledRobustFeatures
	};
	if (deviceFeatures.features.samplerAnisotropy) {
		enabledFeatures.features.samplerAnisotropy = VK_TRUE;
		std::fprintf(stderr, "Enabling device feature: samplerAnisotropy\n");
	}
	if (deviceFeatures.features.imageCubeArray) {
		enabledFeatures.features.imageCubeArray = VK_TRUE;
		std::fprintf(stderr, "Enabling device feature: imageCubeArray\n");
	}
	if (deviceFeatures.features.robustBufferAccess && robustness2Features.robustBufferAccess2) {
		enabledFeatures.features.robustBufferAccess = VK_TRUE;
		enabledRobustFeatures.robustBufferAccess2 = VK_TRUE;
		std::fprintf(stderr, "Enabling device feature: robustBufferAccess\n");
		std::fprintf(stderr, "Enabling device feature: robustBufferAccess2 [robustness2]\n");
	}
	if (robustness2Features.nullDescriptor) {
		enabledRobustFeatures.nullDescriptor = VK_TRUE;
		std::fprintf(stderr, "Enabling device feature: nullDescriptor [robustness2]\n");
	}

	this->deviceFeatures = enabledFeatures;

	VkDeviceCreateInfo deviceInfo{};
	deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	deviceInfo.pNext = &enabledFeatures;
	deviceInfo.queueCreateInfoCount = std::uint32_t(queueInfos.size());
	deviceInfo.pQueueCreateInfos = queueInfos.data();
	deviceInfo.enabledExtensionCount = std::uint32_t(this->enabledDevExtensions.size());
	deviceInfo.ppEnabledExtensionNames = this->enabledDevExtensions.data();
	deviceInfo.pEnabledFeatures = nullptr;

	if (const VkResult res = vkCreateDevice(physicalDevice, &deviceInfo, nullptr, &this->device); VK_SUCCESS != res)
		throw Utils::Error("Unable to create logical device\nvkCreateDevice() returned %s", Utils::toString(res).c_str());
}

void VulkanDevice::createCommandPool() {
	VkCommandPoolCreateInfo poolInfo{};
	poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	poolInfo.queueFamilyIndex = this->graphicsFamilyIndex;
	poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

	if (const VkResult res = vkCreateCommandPool(this->device, &poolInfo, nullptr, &this->cmdPool); VK_SUCCESS != res)
		throw Utils::Error("Unable to create command pool\n vkCreateCommandPool() returned %s", Utils::toString(res).c_str());
}

void VulkanDevice::createDescriptorPool() {
	const VkDescriptorPoolSize pools[] = {
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2048 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 100 },
		{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2048 },
		{ VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 2048 }
	};

	VkDescriptorPoolCreateInfo poolInfo{};
	poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.maxSets = 1024;
	poolInfo.poolSizeCount = sizeof(pools) / sizeof(pools[0]);
	poolInfo.pPoolSizes = pools;
	poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

	if (const VkResult res = vkCreateDescriptorPool(this->device, &poolInfo, nullptr, &this->descPool); VK_SUCCESS != res)
		throw Utils::Error("Unable to create descriptor pool\n vkCreateDescriptorPool() returned %s", Utils::toString(res).c_str());
}

VkPhysicalDevice VulkanDevice::getPhysicalDevice() {
	return this->physicalDevice;
}

VkDevice VulkanDevice::getDevice() const {
	return this->device;
}

const VkPhysicalDeviceProperties& VulkanDevice::getDeviceProperties() const {
	return this->deviceProperties;
}

const VkPhysicalDeviceFeatures2& VulkanDevice::getDeviceFeatures() const {
	return this->deviceFeatures;
}

const std::vector<std::uint32_t>& VulkanDevice::getQueueFamilyIndices() const {
	return this->queueFamilyIndices;
}

std::uint32_t VulkanDevice::getGraphicsFamilyIndex() {
	return this->graphicsFamilyIndex;
}

VkQueue VulkanDevice::getGraphicsQueue() {
	return this->graphicsQueue;
}

std::uint32_t VulkanDevice::getPresentFamilyIndex() {
	return this->presentFamilyIndex;
}

VkQueue VulkanDevice::getPresentQueue() {
	return this->presentQueue;
}

VkCommandPool VulkanDevice::getCmdPool() {
	return this->cmdPool;
}

VkDescriptorPool VulkanDevice::getDescPool() {
	return this->descPool;
}

VkSampleCountFlagBits VulkanDevice::getSampleCount(std::size_t index) {
	return this->possibleSampleCounts[index];
}
