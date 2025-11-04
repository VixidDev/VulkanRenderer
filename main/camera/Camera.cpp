#include "Camera.hpp"

#include <cmath>
#include <algorithm>

#include "../Driver.hpp"
#include "../input/Mouse.hpp"
#include "../vulkan/Swapchain.hpp"

#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

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
	this->invProjection = Cache<glm::mat4>([this]() {
		return glm::inverse(this->getProjection());
	});
	this->view = Cache<glm::mat4>([this]() {
		return glm::lookAt(this->position, this->position + this->frontDir, glm::vec3(0.0f, 1.0f, 0.0f));
	});
	this->invView = Cache<glm::mat4>([this]() {
		return glm::inverse(this->getView());
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
				this->markViewDirty();
				break;
			case GLFW_KEY_S:
				this->position -= distance * this->frontDir;
				this->markViewDirty();
				break;
			case GLFW_KEY_D:
				this->position += glm::normalize(glm::cross(this->frontDir, glm::vec3(0.0f, 1.0f, 0.0f))) * distance;
				this->markViewDirty();
				break;
			case GLFW_KEY_A:
				this->position -= glm::normalize(glm::cross(this->frontDir, glm::vec3(0.0f, 1.0f, 0.0f))) * distance;
				this->markViewDirty();
				break;
			case GLFW_KEY_LEFT_CONTROL:
				this->position -= distance * glm::vec3(0.0f, 1.0f, 0.0f);
				this->markViewDirty();
				break;
			case GLFW_KEY_SPACE:
				this->position += distance * glm::vec3(0.0f, 1.0f, 0.0f);
				this->markViewDirty();
				break;
			case GLFW_KEY_P:
				this->animating = !this->animating;
				break;
			}
		}
	}

	if (this->animating) {
		this->playAnimation(timeDelta);
		//return;
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
	this->markViewDirty();
 }

void Camera::markProjectionDirty() {
	this->projection.markDirty();
	this->invProjection.markDirty();
}

void Camera::markViewDirty() {
	this->view.markDirty();
	this->invView.markDirty();
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

glm::mat4 Camera::getInvProjection() {
	return this->invProjection.get();
}

glm::mat4 Camera::getView() {
	return this->view.get();
}

glm::mat4 Camera::getInvView() {
	return this->invView.get();
}

float& Camera::getSensitivity() {
	return this->sensitivity;
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

constexpr glm::vec3 animatedPoints[17] = {
	glm::vec3(-8.3f,  5.0f, -5.4f),  // Outside
	glm::vec3( 0.0f,  2.1f, -12.0f), // In front of angel
	glm::vec3( 4.2f,  2.1f, -13.7f), // Slight angle of angel
	glm::vec3( 6.2f,  2.1f, -16.7f), // Side on angel
	glm::vec3( 0.0f,  2.1f, -20.7f), // Looking down hallway
	glm::vec3( 0.0f,  2.1f, -34.2f), // End of hallway
	glm::vec3(-0.6f,  1.3f, -36.2f), // Turning left
	glm::vec3(-2.5f,  1.0f, -37.1f), // Facing left
	glm::vec3(-5.4f,  0.1f, -37.4f), // Turning right
	glm::vec3(-5.5f, -0.3f, -38.5f), // Facing right
	glm::vec3(-5.5f, -0.9f, -43.6f), // Along side angel
	glm::vec3(-5.5f, -0.9f, -50.6f), // Looking at angel
	glm::vec3(-5.0f, -0.9f, -55.8f), // Looking at back hallway
	glm::vec3(-4.4f, -0.9f, -59.3f), // Turning into hallway
	glm::vec3(-1.1f, -0.9f, -60.2f), // Turning into hallway 2
	glm::vec3( 0.0f, -1.0f, -62.5f), // In back hallway
	glm::vec3( 0.0f, -1.0f, -81.1f), // End of back hallway
};

constexpr glm::vec2 animatedAngles[17] = {
	// x = yaw, y = pitch
	glm::vec2(-55.0f,  -21.0f), // Outside
	glm::vec2(-90.0f,   0.0f),  // In front of angel
	glm::vec2(-135.0f,  2.0f),  // Slight angle of angel
	glm::vec2(-170.0f,  2.0f),  // Side on angel
	glm::vec2(-90.0f,   0.0f),  // Looking down hallway
	glm::vec2(-90.0f,   0.0f),  // End of hallway
	glm::vec2(-138.0f, -11.0f), // Turning left
	glm::vec2( 180.0f, -21.1f), // Facing left
	glm::vec2(-147.0f, -13.1f), // Turning right
	glm::vec2(-90.0f,  -7.6f),  // Facing right
	glm::vec2(-85.0f,  -6.9f),  // Along side angel
	glm::vec2( 17.0f,  -3.0f),  // Looking at angel
	glm::vec2(-60.0f,  -5.9f),  // Looking at back hallway
	glm::vec2(-20.0f,  -1.6f),  // Turning into hallway
	glm::vec2(-51.0f,   0.0f),  // Turning into hallway 2
	glm::vec2(-90.0f,   0.0f),  // In back hallway
	glm::vec2(-90.0f,   0.0f),  // End of back hallway
};

constexpr float animationTimings[17] = {
	0.00000f,  1.86974f,  2.64034f,  3.25356f, 
	4.50842f,  6.80443f,  7.18473f,  7.54591f, 
	8.06485f,  8.26464f,  9.13800f,  10.32852f, 
	11.21699f, 11.82093f, 12.40268f, 12.83662f, 
	16.00000f
};

void Camera::playAnimation(float timeDelta) {
	if (!this->animating) return;

	for (int i = 0; i < 16; i++) {
		float firstKeyframe = animationTimings[i] * 2.0f;
		float secondKeyframe = animationTimings[i + 1] * 2.0f;

		if (this->animationTimer < firstKeyframe || this->animationTimer > secondKeyframe)
			continue;

		float interp = std::max(0.0f, this->animationTimer - firstKeyframe) / (secondKeyframe - firstKeyframe);

		glm::vec3 posA = animatedPoints[i];
		glm::vec3 posB = animatedPoints[i + 1];
		this->position = glm::mix(posA, posB, interp);

		//auto quatFromPitchYaw = [](float pitch, float yaw) {
		//	glm::quat qPitch = glm::angleAxis(glm::radians(pitch), glm::vec3(0, 1, 0));
		//	glm::quat qYaw = glm::angleAxis(glm::radians(yaw), glm::vec3(1, 0, 0));
		//	return qPitch * qYaw;
		//};

		//glm::quat qA = quatFromPitchYaw(animatedAngles[i].y, animatedAngles[i].x);
		//glm::quat qB = quatFromPitchYaw(animatedAngles[i + 1].y, animatedAngles[i + 1].x);

		//glm::quat qInterp = glm::slerp(qA, qB, interp);
		//glm::vec3 eulerAngles = glm::eulerAngles(qInterp);

		//this->pitch = eulerAngles.y;
		//this->yaw = eulerAngles.x;

		//glm::vec3 newDir{};
		//newDir.x = std::cos(this->yaw) * std::cos(this->pitch);
		//newDir.y = std::sin(this->pitch);
		//newDir.z = std::sin(this->yaw) * std::cos(this->pitch);
		//this->frontDir = glm::normalize(newDir);
		this->markViewDirty();

		//this->pitch = glm::degrees(this->pitch);
		//this->yaw = glm::degrees(this->yaw);
	}

	this->animationTimer += timeDelta;

	if (this->animationTimer > this->animationDuration * 2.0f) {
		this->animating = false;
		this->animationTimer = 0.0f;
	}
}