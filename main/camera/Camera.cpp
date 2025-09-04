#include "Camera.hpp"

#include <cmath>
#include <algorithm>

#include "../Driver.hpp"
#include "../input/Mouse.hpp"

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

Camera::Camera(float fov, float nearPlane, float farPlane, glm::vec3 position, glm::vec3 frontDir) :
	fov(fov), nearPlane(nearPlane), farPlane(farPlane), position(position), frontDir(frontDir) 
{}

void Camera::update(GLFWwindow* window, float timeDelta) {
	if (glfwGetInputMode(window, GLFW_CURSOR) != GLFW_CURSOR_DISABLED)
		return;

	UserState* state = static_cast<UserState*>(glfwGetWindowUserPointer(window));

	for (const auto& [key, buttonState] : state->keyState) {
		if (buttonState == ButtonState::PRESSED || buttonState == ButtonState::HELD) {
			float distance = 5.0f * timeDelta;

			switch (key) {
			case GLFW_KEY_W:
				this->position += distance * this->frontDir;
				break;
			case GLFW_KEY_S:
				this->position -= distance * this->frontDir;
				break;
			case GLFW_KEY_D:
				this->position += glm::normalize(glm::cross(this->frontDir, glm::vec3(0.0f, 1.0f, 0.0f))) * distance;
				break;
			case GLFW_KEY_A:
				this->position -= glm::normalize(glm::cross(this->frontDir, glm::vec3(0.0f, 1.0f, 0.0f))) * distance;
				break;
			}
		}
	}

	float xOffset, yOffset;

	if (state->firstClick) {
		int width, height;
		glfwGetFramebufferSize(window, &width, &height);
		glfwSetCursorPos(window, width / 2.0f, height / 2.0f);
		Mouse::setX(width / 2.0f);
		Mouse::setY(height / 2.0f);
		this->lastX = width / 2.0f;
		this->lastY = height / 2.0f;
		state->firstClick = false;
	}

	xOffset = Mouse::getX() - this->lastX;
	yOffset = this->lastY - Mouse::getY();

	if (xOffset != 0.0f || yOffset != 0.0f) {
		int a = 1.0f;
	}

	this->lastX = Mouse::getX();
	this->lastY = Mouse::getY();

	xOffset *= this->sensitivity;
	yOffset *= this->sensitivity;

	this->yaw += xOffset;
	this->pitch += yOffset;

	if (this->pitch > 89.9f)
		this->pitch = 89.9f;
	if (this->pitch < -89.9f)
		this->pitch = -89.9f;

	if (this->yaw > 180.0f)
		this->yaw = -180.0f;
	if (this->yaw < -180.0f)
		this->yaw = 180.0f;

	glm::vec3 newDir{};
	newDir.x = std::cos(glm::radians(this->yaw)) * std::cos(glm::radians(this->pitch));
	newDir.y = std::sin(glm::radians(this->pitch));
	newDir.z = std::sin(glm::radians(this->yaw)) * std::cos(glm::radians(this->pitch));
	this->frontDir = glm::normalize(newDir);
 }

float Camera::getFov() {
	return this->fov;
}

float Camera::getNearPlane() {
	return this->nearPlane;
}

float Camera::getFarPlane() {
	return this->farPlane;
}

glm::vec3 Camera::getPosition() {
	return this->position;
}

glm::vec3 Camera::getFrontDir() {
	return this->frontDir;
}

float Camera::getYaw() {
	return this->yaw;
}

float Camera::getPitch() {
	return this->pitch;
}
