#include "Camera.hpp"

#include <cmath>
#include <algorithm>

#include "../Driver.hpp"
#include "../input/Mouse.hpp"
#include "../vulkan/Swapchain.hpp"

#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>

Camera::Camera(Swapchain* swapchain, float fov, float nearPlane, float farPlane, glm::vec3 position, glm::vec3 frontDir) :
	swapchain(swapchain), fov(fov), nearPlane(nearPlane), farPlane(farPlane), position(position), frontDir(frontDir)
{
	this->projection = Cache<glm::mat4>([this]() {
		float width = static_cast<float>(this->swapchain->getExtent().width);
		float height = static_cast<float>(this->swapchain->getExtent().height);
		const float aspectRatio = width / height;

		glm::mat4 mat = glm::perspective(glm::radians(this->fov), aspectRatio, this->nearPlane, this->farPlane);
		mat[1][1] *= -1.0f;
		return mat;
	});
	this->view = Cache<glm::mat4>([this]() {
		return glm::lookAt(this->position, this->position + this->frontDir, glm::vec3(0.0f, 1.0f, 0.0f));
	});
}

void Camera::update(GLFWwindow* glfwWindow, float timeDelta) {
	// Return if mouse isnt focused
	if (glfwGetInputMode(glfwWindow, GLFW_CURSOR) != GLFW_CURSOR_DISABLED)
		return;

	UserState* state = static_cast<UserState*>(glfwGetWindowUserPointer(glfwWindow));

	for (const auto& [key, buttonState] : state->keyState) {
		if (buttonState == ButtonState::PRESSED || buttonState == ButtonState::HELD) {
			float distance = 5.0f * timeDelta;

			if (state->modifiers & GLFW_MOD_SHIFT) distance *= 3.0f;

			switch (key) {
			case GLFW_KEY_W:
				this->position += distance * this->frontDir;
				this->view.markDirty();
				break;
			case GLFW_KEY_S:
				this->position -= distance * this->frontDir;
				this->view.markDirty();
				break;
			case GLFW_KEY_D:
				this->position += glm::normalize(glm::cross(this->frontDir, glm::vec3(0.0f, 1.0f, 0.0f))) * distance;
				this->view.markDirty();
				break;
			case GLFW_KEY_A:
				this->position -= glm::normalize(glm::cross(this->frontDir, glm::vec3(0.0f, 1.0f, 0.0f))) * distance;
				this->view.markDirty();
				break;
			case GLFW_KEY_LEFT_CONTROL:
				this->position -= distance * glm::vec3(0.0f, 1.0f, 0.0f);
				this->view.markDirty();
				break;
			case GLFW_KEY_SPACE:
				this->position += distance * glm::vec3(0.0f, 1.0f, 0.0f);
				this->view.markDirty();
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

	// If both xOffset and yOffset is 0, mouse hasn't moved
	if (xOffset == 0.0f && yOffset == 0.0f) return;

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

	// If xOffset or yOffset is non-zero, it is very likely the front dir will change
	// so we compute it always if either are non-zero
	glm::vec3 newDir{};
	newDir.x = std::cos(glm::radians(this->yaw)) * std::cos(glm::radians(this->pitch));
	newDir.y = std::sin(glm::radians(this->pitch));
	newDir.z = std::sin(glm::radians(this->yaw)) * std::cos(glm::radians(this->pitch));
	this->frontDir = glm::normalize(newDir);
	this->view.markDirty();
 }

void Camera::markProjectionDirty() {
	this->projection.markDirty();
}

void Camera::markViewDirty() {
	this->view.markDirty();
}

float& Camera::getFov() {
	return this->fov;
}

float& Camera::getNearPlane() {
	return this->nearPlane;
}

float& Camera::getFarPlane() {
	return this->farPlane;
}

glm::vec3 Camera::getPosition() {
	return this->position;
}

glm::vec3 Camera::getFrontDir() {
	return this->frontDir;
}

glm::mat4 Camera::getProjection() {
	return this->projection.get();
}

glm::mat4 Camera::getView() {
	return this->view.get();
}

std::array<glm::vec4, 8> Camera::getFrustumCorners() {
	assert(this->projection.get() != glm::mat4{} && "Camera projection matrix must be initialised before getting frustum corners!");
	assert(this->view.get() != glm::mat4{} && "Camera view matrix must be initialised before getting frustum corners!");

	glm::mat4 inverseViewProj = glm::inverse(this->projection.get() * this->view.get());

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
