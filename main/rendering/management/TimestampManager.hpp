#pragma once

#include <vector>
#include <map>
#include <string>
#include <optional>

#include "../vulkan/objects/VkObjects.hpp"

struct VulkanContext;

struct IndexReference {
	int start = -1;
	int end = -1;
};

using TimestampReferences = std::vector<std::pair<std::string, IndexReference>>;

class TimestampManager {
public:
	TimestampManager() = default;
	TimestampManager(VulkanContext* context);

	void resetGPUQueryPool();
	void flushCPUTimestamps();

	void writeGPUTimestamp(std::string reference, VkPipelineStageFlagBits stageFlag);
	void writeCPUTimestamp(std::string reference, std::uint64_t timestamp);

	void readBackGPUTimestamps();

	std::optional<std::uint64_t> getGPUTimestamp(int index);
	std::optional<std::uint64_t> getCPUTimestamp(int index);

	TimestampReferences& getGPUTimestampReferences();
	TimestampReferences& getCPUTimestampReferences();

private:
	VulkanContext* context = nullptr;

	vk::QueryPool gpuQueryPool;
	std::uint32_t gpuQueryCounter{};
	int cpuQueryCounter{};

	std::vector<std::uint64_t> gpuTimestamps;
	std::vector<std::uint64_t> cpuTimestamps;

	TimestampReferences gpuTimestampReferences;
	TimestampReferences cpuTimestampReferences;

	TimestampReferences lastFrameCpuTimestampReferences;
	std::vector<std::uint64_t> lastFrameCpuTimestamps;
};