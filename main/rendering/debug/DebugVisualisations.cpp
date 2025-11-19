#include "DebugVisualisations.hpp"

#include "../Renderer.hpp"
#include "../Driver.hpp"
#include "../RendererUtils.hpp"
#include "../models/ModelLoader.hpp"

namespace Debug {

	vk::Buffer lightVolumeDebugBuffer;

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
			if (lights[i].getLightType() == LightType::POINT) {
				count++;

				InstanceData data = {
					.translation = lights[i].getPosition(),
					.scale = std::sqrt(lights[i].getIntensity() / 0.025f),
					.colour = glm::vec4(0.24f, 0.78f, 0.48f, 0.2f)
				};
				instanceData.emplace_back(data);
			}
		}

		vk::Buffer& lightVolumesBuffer = getLightVolumesDebugBuffer(renderer, instanceData);

		VkUtils::beginCmdLabel(RendererUtils::getCommandBuffer(), "Debug Light Volumes");
		RendererUtils::beginRenderPass(renderPass, framebuffer, imageIndex);
		RendererUtils::bindGraphicPipeline(pipeline->getHandle());
		RendererUtils::bindGraphicDescriptorSets(pipelineLayout->getHandle(), 0, 1, &mvpDescriptorSet);
		RendererUtils::drawDebugMeshInstanced(renderer->getDebugSphere(), count, lightVolumesBuffer);
		RendererUtils::endRenderPass();
		VkUtils::endCmdLabel(RendererUtils::getCommandBuffer());
	}

	vk::Buffer& getLightVolumesDebugBuffer(Renderer* renderer, std::vector<InstanceData>& instanceData) {
		if (lightVolumeDebugBuffer) return lightVolumeDebugBuffer;

		VulkanWindow& window = *renderer->getContext().window;
		VulkanAllocator& allocator = *renderer->getContext().allocator;

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

		// (should mapToGPU be only in the ModelLoader?)
		ModelLoader::mapToGPU(allocator, lightVolumeDebugBuffer, staging, instanceData);

		VkCommandBuffer uploadCmdBuf = VkUtils::createCommandBuffer(window, window.getDevice()->getCmdPool());
		VkUtils::beginCommandBuffer(uploadCmdBuf);

		// If a vector is empty, the copy method does nothing
		ModelLoader::copyToGPU(uploadCmdBuf, lightVolumeDebugBuffer, staging, instanceData);

		VkUtils::endAndSubmitCommandBuffer(window, uploadCmdBuf);

		return lightVolumeDebugBuffer;
	}

	void destroyBuffers() {
		lightVolumeDebugBuffer.~Buffer();
	}

}