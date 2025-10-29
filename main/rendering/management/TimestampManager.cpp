#include "TimestampManager.hpp"

#include <stdexcept>

#include "../RendererUtils.hpp"

TimestampManager::TimestampManager(VulkanContext* context) : context(context) {
	this->gpuQueryPool = VkUtils::createQueryPool(*this->context->window, VK_QUERY_TYPE_TIMESTAMP, 100);
	this->gpuTimestamps.resize(100);
}

void TimestampManager::resetGPUQueryPool() {
	if (!this->recordGPUTimestamps) return;

	RendererUtils::resetQueryPool(this->gpuQueryPool, 0, 100);
	this->gpuQueryCounter = 0;
	this->gpuTimestampReferences.clear();
}

void TimestampManager::flushCPUTimestamps() {
	if (!this->recordCPUTimestamps) return;

	// Save references and timestamps to be read by ImGui so we can reset the actual counters
	this->lastFrameCpuTimestampReferences = this->cpuTimestampReferences;
	this->lastFrameCpuTimestamps = this->cpuTimestamps;

	this->cpuQueryCounter = 0;
	this->cpuTimestampReferences.clear();
	this->cpuTimestamps.clear();
}

void TimestampManager::clearCPUTimestamps() {
	if (!this->recordCPUTimestamps) return;

	this->cpuQueryCounter = 0;
	this->cpuTimestampReferences.clear();
	this->cpuTimestamps.clear();
	this->lastFrameCpuTimestampReferences.clear();
	this->lastFrameCpuTimestamps.clear();
}

void TimestampManager::writeGPUTimestamp(std::string reference, VkPipelineStageFlagBits stageFlag) {
	if (!this->recordGPUTimestamps) return;

	for (auto& [name, indexReference] : this->gpuTimestampReferences) {
		if (name == reference) {
			if (indexReference.end != -1) {
				std::fprintf(stderr, "TimestampManager: gpuTimestampReferences already contains reference to %s. Ignoring this write.\n", reference.c_str());
				return;
			}

			indexReference.end = this->gpuQueryCounter;
			RendererUtils::writeTimestamp(stageFlag, this->gpuQueryPool, this->gpuQueryCounter);
			return;
		}
	}

	this->gpuTimestampReferences.emplace_back(reference, IndexReference{ static_cast<int>(this->gpuQueryCounter), -1 });
	RendererUtils::writeTimestamp(stageFlag, this->gpuQueryPool, this->gpuQueryCounter);
}

void TimestampManager::writeCPUTimestamp(std::string reference) {
	if (!this->recordCPUTimestamps) return;

	std::uint64_t timestamp = std::chrono::duration_cast<Nanoseconds>(Clock::now().time_since_epoch()).count();

	for (auto& [name, indexReference] : this->cpuTimestampReferences) {
		if (name == reference) {
			if (indexReference.end != -1) {
				std::fprintf(stderr, "TimestampManager: cpuTimestampReferences already contains reference to %s. Ignoring this write.\n", reference.c_str());
				return;
			}

			this->cpuTimestamps.emplace_back(timestamp);
			indexReference.end = this->cpuQueryCounter;
			this->cpuQueryCounter++;
			return;
		}
	}

	this->cpuTimestamps.emplace_back(timestamp);
	this->cpuTimestampReferences.emplace_back(reference, IndexReference{ this->cpuQueryCounter, -1 });
	this->cpuQueryCounter++;
}

void TimestampManager::readBackGPUTimestamps() {
	if (!this->recordGPUTimestamps) return;

	VkUtils::getQueryPoolResults(
		*this->context->window,
		this->gpuQueryPool,
		this->gpuTimestamps,
		this->gpuQueryCounter
	);
}

std::optional<std::uint64_t> TimestampManager::getGPUTimestamp(int index) {
	std::optional<std::uint64_t> res{};

	try {
		res = this->gpuTimestamps.at(index);
	} catch (const std::out_of_range&) {
		std::fprintf(stderr, "Index: %d is out of range for gpuTimestamps!\n", index);
	}

	return res;
}

std::optional<std::uint64_t> TimestampManager::getCPUTimestamp(int index) {
	std::optional<std::uint64_t> res{};

	try {
		res = this->lastFrameCpuTimestamps.at(index);
	} catch (const std::out_of_range&) {
		std::fprintf(stderr, "Index: %d is out of range for cpuTimestamps!\n", index);
	}

	return res;
}

TimestampReferences& TimestampManager::getGPUTimestampReferences() {
	return this->gpuTimestampReferences;
}

TimestampReferences& TimestampManager::getCPUTimestampReferences() {
	return this->lastFrameCpuTimestampReferences;
}
