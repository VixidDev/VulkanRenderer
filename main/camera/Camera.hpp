#pragma once

#include <array>
#include <glm/glm.hpp>

#include "Cache.hpp"

class Swapchain;
struct GLFWwindow;

class Camera {
public:
	Camera() = default;
	Camera(Swapchain* swapchain, float fov, float nearPlane, float farPlane, glm::vec3 position, glm::vec3 frontDir);
	~Camera() = default;

	// Delete move and copy since projection
	// and view Cache capture a reference during
	// construction
	Camera(const Camera&) = delete;
	Camera& operator=(const Camera&) = delete;
	Camera(Camera&&) = delete;
	Camera& operator=(Camera&&) = delete;

	void update(GLFWwindow* window, float timeDelta);

	void markProjectionDirty();
	void markViewDirty();

	float& getFov();
	float& getNearPlane();
	float& getFarPlane();
	glm::vec3 getPosition();
	glm::vec3 getFrontDir();

	glm::mat4 getProjection();
	glm::mat4 getInvProjection();
	glm::mat4 getView();
	glm::mat4 getInvView();

	std::array<glm::vec4, 8> getFrustumCorners();

	float getYaw();
	float getPitch();
private:
	Swapchain* swapchain;

	float fov;
	float nearPlane;
	float farPlane;
	glm::vec3 position;
	glm::vec3 frontDir;

	Cache<glm::mat4> projection;
	Cache<glm::mat4> invProjection;
	Cache<glm::mat4> view;
	Cache<glm::mat4> invView;

	float sensitivity = 0.25f;

	float yaw = -90.0f;
	float pitch = 0.0f;
	float lastX = 0.0f;
	float lastY = 0.0f;
};