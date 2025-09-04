#pragma once

#include <glm/vec3.hpp>

struct GLFWwindow;

class Camera {
public:
	Camera() = default;
	Camera(float fov, float nearPlane, float farPlane, glm::vec3 position, glm::vec3 frontDir);

	~Camera() = default;

	void update(GLFWwindow* window, float timeDelta);

	float getFov();
	float getNearPlane();
	float getFarPlane();
	glm::vec3 getPosition();
	glm::vec3 getFrontDir();

	float getYaw();
	float getPitch();
private:
	float fov;
	float nearPlane;
	float farPlane;
	glm::vec3 position;
	glm::vec3 frontDir;

	float sensitivity = 0.25f;

	float yaw = -90.0f;
	float pitch = 0.0f;
	float lastX = 0.0f;
	float lastY = 0.0f;
};