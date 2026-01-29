#include "DebugVisualisations.hpp"

#include "../Renderer.hpp"
#include "../Driver.hpp"
#include "../RendererUtils.hpp"
#include "../models/ModelLoader.hpp"

namespace Debug {

	vk::Buffer lightVolumeDebugBuffer;
	std::optional<std::vector<InstanceData>> oldData;

	void renderDebugLightVolumes(Renderer* renderer, uint32_t imageIndex) {
		RenderPass* renderPass = renderer->getRenderPass("debugShapes");
		Framebuffer* framebuffer = renderer->getFramebuffer("debugShapes");
		Pipeline* pipeline = renderer->getPipeline("debugShapes");
		PipelineLayout* pipelineLayout = renderer->getPipelineLayout("lineDebug");
		VkDescriptorSet mvpDescriptorSet = RendererUtils::getDescriptorSetHandle(renderer->getDescriptorSet("mvp"));

		std::vector<Light>& lights = renderer->getDriver()->getLights();
		std::vector<InstanceData> instanceData;
		uint32_t count = 0;
		for (size_t i = 0; i < lights.size(); i++) {
			const Light& light = lights[i];

			if (light.getLightType() == LightType::POINT) {
				if (!light.isEnabled()) continue;
				count++;

				InstanceData data = {
					.translation = light.getPosition(),
					.scale = light.getRadius(),
					.colour = glm::vec4(0.24f, 0.78f, 0.48f, 0.2f)
				};
				instanceData.emplace_back(data);
			}
		}

		// If no lights are visible, just return
		if (count <= 0) return;

		// Check if we need to force update the GPU buffer
		bool forceUpdate = shouldUpdateBuffer(instanceData);		
		vk::Buffer& lightVolumesBuffer = getLightVolumesDebugBuffer(renderer, instanceData, forceUpdate);

		VkUtils::beginCmdLabel(RendererUtils::getCommandBuffer(), "Debug Light Volumes");
		RendererUtils::beginRenderPass(renderPass, framebuffer, imageIndex);
		RendererUtils::bindGraphicPipeline(pipeline->getHandle());
		RendererUtils::bindGraphicDescriptorSets(pipelineLayout->getHandle(), 0, 1, &mvpDescriptorSet);
		RendererUtils::drawDebugMeshInstanced(renderer->getDebugSphere(), count, lightVolumesBuffer);
		RendererUtils::endRenderPass();
		VkUtils::endCmdLabel(RendererUtils::getCommandBuffer());
	}

	vk::Buffer& getLightVolumesDebugBuffer(Renderer* renderer, std::vector<InstanceData>& instanceData, bool forceUpdate) {
		if (lightVolumeDebugBuffer && !forceUpdate) return lightVolumeDebugBuffer;


		VulkanWindow& window = *renderer->getContext().window;
		VulkanAllocator& allocator = *renderer->getContext().allocator;

		// Given that we only end up here when a debug feature is enabled,
		// its ok we use vkDeviceWaitIdle
		vkDeviceWaitIdle(window.getDevice()->getDevice());

		lightVolumeDebugBuffer = vk::Buffer::createBuffer(
			allocator,
			sizeof(InstanceData) * instanceData.size(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			0,
			VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
		);

		vk::Buffer staging = vk::Buffer::createBuffer(
			allocator,
			sizeof(InstanceData) * instanceData.size(),
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
		);

		VkUtils::setObjectName(window.getDevice()->getDevice(), VK_OBJECT_TYPE_BUFFER, (uint64_t)lightVolumeDebugBuffer.get(), "Light Volume Debug Buffer");
		VkUtils::setObjectName(window.getDevice()->getDevice(), VK_OBJECT_TYPE_BUFFER, (uint64_t)staging.get(), "Light Volume Debug Staging Buffer");

		// (should mapToGPU be only in the ModelLoader?)
		ModelLoader::mapToGPU(allocator, lightVolumeDebugBuffer, staging, instanceData);

		VkCommandBuffer uploadCmdBuf = VkUtils::createCommandBuffer(window, window.getDevice()->getCmdPool());
		VkUtils::setObjectName(window.getDevice()->getDevice(), VK_OBJECT_TYPE_COMMAND_BUFFER, (uint64_t)uploadCmdBuf, "Temp Upload Command Buffer (Light Volumes Debug Buffer Upload)");
		VkUtils::beginCommandBuffer(uploadCmdBuf);

		// If a vector is empty, the copy method does nothing
		ModelLoader::copyToGPU(uploadCmdBuf, lightVolumeDebugBuffer, staging, instanceData);

		VkUtils::endAndSubmitCommandBuffer(window, uploadCmdBuf);

		return lightVolumeDebugBuffer;
	}

	bool shouldUpdateBuffer(std::vector<InstanceData>& instanceData) {
		if (!oldData.has_value()) oldData = instanceData;

		// Quick size check
		if (oldData.value().size() != instanceData.size()) return true;

		// If both are same size, check positions of both
		for (size_t i = 0; i < instanceData.size(); i++) {
			if (oldData.value()[i].translation != instanceData[i].translation) return true;
		}

		// If both are same, don't need to update
		return false;
	}

	void destroyBuffers() {
		lightVolumeDebugBuffer.~Buffer();
	}

}