#pragma once

#include <array>
#include <glm/glm.hpp>

class Swapchain;
struct GLFWwindow;

class Camera {
public:
	Camera() = default;
	Camera(Swapchain* swapchain, float fov, float nearPlane, float farPlane, glm::vec3 position, glm::vec3 frontDir);

	~Camera() = default;

	void update(GLFWwindow* window, float timeDelta);

	float& getFov();
	float& getNearPlane();
	float& getFarPlane();
	glm::vec3 getPosition();
	glm::vec3 getFrontDir();

	glm::mat4 getProjectionMat();
	glm::mat4 getViewMat();

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

	glm::mat4 projection{};
	glm::mat4 view{};

	float sensitivity = 0.25f;

	float yaw = -90.0f;
	float pitch = 0.0f;
	float lastX = 0.0f;
	float lastY = 0.0f;
};