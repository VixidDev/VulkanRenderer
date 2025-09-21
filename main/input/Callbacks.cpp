#include "Callbacks.hpp"

#include "../Driver.hpp"
#include "Mouse.hpp"

void Callbacks::onKey(GLFWwindow* window, int key, int scanCode, int action, int modifiers) {
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GLFW_TRUE);

	UserState* state = static_cast<UserState*>(glfwGetWindowUserPointer(window));

	switch (action) {
	case GLFW_PRESS:
	{
		state->keyState[key] = ButtonState::PRESSED;
		break;
	}
	case GLFW_RELEASE:
	{
		state->keyState[key] = ButtonState::RELEASED;
		break;
	}
	case GLFW_REPEAT:
	{
		state->keyState[key] = ButtonState::HELD;
		break;
	}
	}

	state->modifiers = modifiers;
}

void Callbacks::onMouseButton(GLFWwindow* window, int button, int action, int modifiers) {
	UserState* state = static_cast<UserState*>(glfwGetWindowUserPointer(window));

	if (action == GLFW_PRESS && button == GLFW_MOUSE_BUTTON_2) {
		if (glfwGetInputMode(window, GLFW_CURSOR) == GLFW_CURSOR_NORMAL) {
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
			state->firstClick = true;
		} else {
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		}
	}

	switch (action) {
	case GLFW_PRESS:
		state->mouseState[button] = ButtonState::PRESSED;
		break;
	case GLFW_RELEASE:
		state->mouseState[button] = ButtonState::RELEASED;
		break;
	case GLFW_REPEAT:
		state->mouseState[button] = ButtonState::HELD;
		break;
	}

	state->modifiers = modifiers;
}

void Callbacks::onMouseMove(GLFWwindow* window, double x, double y) {
	if (glfwGetInputMode(window, GLFW_CURSOR) != GLFW_CURSOR_DISABLED)
		return;

	Mouse::setX(static_cast<float>(x));
	Mouse::setY(static_cast<float>(y));
}