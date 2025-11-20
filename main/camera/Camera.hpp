#pragma once

#include <array>
#include <glm/glm.hpp>

#include "Cache.hpp"
#include "../rendering/lights/Light.hpp"

class Swapchain;
struct GLFWwindow;

struct FrustumPlane {
	glm::vec3 normal;
	float d;
};

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

	float& getFov() { return this->fov; }
	float& getNearPlane() { return this->nearPlane; }
	float& getFarPlane() { return this->nearPlane; }
	float& getSensitivity() { return this->sensitivity; }

	glm::vec3 getPosition() const { return this->position; }
	glm::vec3 getFrontDir() const { return this->frontDir; }
	float getYaw() const { return this->yaw; }
	float getPitch() const { return this->pitch; }

	glm::mat4 getProjection() { return this->projection.get(); }
	glm::mat4 getInvProjection() { return this->invProjection.get(); }
	glm::mat4 getView() { return this->view.get(); }
	glm::mat4 getInvView() { return this->invView.get(); }

	std::array<glm::vec4, 8> getFrustumCorners();
	std::array<FrustumPlane, 6> getFrustumPlanes();

	bool lightInterectsFrustum(Light& light);
private:
	void playAnimation(float timeDelta);

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

	bool animating = false;
	float animationDuration = 16.0f;
	float animationTimer = 0.0f;
};