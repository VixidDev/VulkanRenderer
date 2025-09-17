#include "Camera.hpp"

#include <cmath>
#include <algorithm>

#include "../Driver.hpp"
#include "../input/Mouse.hpp"
#include "../vulkan/VulkanWindow.hpp"

#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>

Camera::Camera(VulkanWindow* window, float fov, float nearPlane, float farPlane, glm::vec3 position, glm::vec3 frontDir) :
	window(window), fov(fov), nearPlane(nearPlane), farPlane(farPlane), position(position), frontDir(frontDir) 
{
	float width = static_cast<float>(this->window->swapchainExtent.width);
	float height = static_cast<float>(this->window->swapchainExtent.height);
	const float aspectRatio = width / height;

	// Initialise the transformation matrices for the first time
	this->projection = glm::perspective(
		glm::radians(this->fov),
		aspectRatio,
		this->nearPlane,
		this->farPlane
	);
	this->projection[1][1] *= -1.0f;
	this->view = glm::lookAt(
		this->position,
		this->position + this->frontDir,
		glm::vec3(0.0f, 1.0f, 0.0f)
	);
}

void Camera::update(GLFWwindow* glfwWindow, float timeDelta) {
	if (glfwGetInputMode(glfwWindow, GLFW_CURSOR) != GLFW_CURSOR_DISABLED)
		return;

	float width = static_cast<float>(window->swapchainExtent.width);
	float height = static_cast<float>(window->swapchainExtent.height);
	const float aspectRatio = width / height;

	this->projection = glm::perspective(
		glm::radians(this->fov),
		aspectRatio,
		this->nearPlane,
		this->farPlane
	);
	this->projection[1][1] *= -1.0f;
	this->view = glm::lookAt(
		this->position,
		this->position + this->frontDir,
		glm::vec3(0.0f, 1.0f, 0.0f)
	);

	UserState* state = static_cast<UserState*>(glfwGetWindowUserPointer(glfwWindow));

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
		glfwGetFramebufferSize(glfwWindow, &width, &height);
		glfwSetCursorPos(glfwWindow, width / 2.0f, height / 2.0f);
		Mouse::setX(width / 2.0f);
		Mouse::setY(height / 2.0f);
		this->lastX = width / 2.0f;
		this->lastY = height / 2.0f;
		state->firstClick = false;
	}

	xOffset = Mouse::getX() - this->lastX;
	yOffset = this->lastY - Mouse::getY();

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

glm::mat4 Camera::getProjectionMat() {
	return this->projection;
}

glm::mat4 Camera::getViewMat() {
	return this->view;
}

std::array<glm::vec4, 8> Camera::getFrustumCorners() {
	assert(this->projection != glm::mat4{} && "Camera projection matrix must be initialised before getting frustum corners!");
	assert(this->view != glm::mat4{} && "Camera view matrix must be initialised before getting frustum corners!");

	glm::mat4 inverseViewProj = glm::inverse(this->projection * this->view);

	std::vector<glm::vec3> ndcCorners = {
		// Near plane corners
		{-1, -1, -1}, {1, -1, -1}, {1,  1, -1}, {-1,  1, -1},
		// Far plane corners
		{-1, -1,  1}, {1, -1,  1}, {1,  1,  1}, {-1,  1,  1}
	};

	std::array<glm::vec4, 8> frustumCorners{};
	for (std::size_t i = 0; i < ndcCorners.size(); i++) {
		glm::vec4 worldSpaceCorner = inverseViewProj * glm::vec4(ndcCorners[i], 1.0f);
		worldSpaceCorner /= worldSpaceCorner.w;

		frustumCorners[i] = worldSpaceCorner;
	}

	return frustumCorners;
}

float Camera::getYaw() {
	return this->yaw;
}

float Camera::getPitch() {
	return this->pitch;
}
