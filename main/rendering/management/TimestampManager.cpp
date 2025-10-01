#include "TimestampManager.hpp"

#include <stdexcept>

#include "../RendererUtils.hpp"

TimestampManager::TimestampManager(VulkanContext* context) : context(context) {
	this->gpuQueryPool = VkUtils::createQueryPool(*this->context->window, VK_QUERY_TYPE_TIMESTAMP, 100);
	this->gpuTimestamps.resize(100);
}

void TimestampManager::resetGPUQueryPool() {
	RendererUtils::resetQueryPool(this->gpuQueryPool, 0, 100);
	this->gpuQueryCounter = 0;
	this->gpuTimestampReferences.clear();
}

void TimestampManager::writeGPUTimestamp(std::string reference, VkPipelineStageFlagBits stageFlag) {
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

void TimestampManager::readBackGPUTimestamps() {
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
		res = this->cpuTimestamps.at(index);
	} catch (const std::out_of_range&) {
		std::fprintf(stderr, "Index: %d is out of range for cpuTimestamps!\n", index);
	}

	return res;
}

TimestampReferences& TimestampManager::getGPUTimestampReferences() {
	return this->gpuTimestampReferences;
}

TimestampReferences& TimestampManager::getCPUTimestampReferences() {
	return this->cpuTimestampReferences;
}
