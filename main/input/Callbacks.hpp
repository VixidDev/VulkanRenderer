#pragma once

#include <map>

#include <GLFW/glfw3.h>

struct GLFWwindow;

enum class ButtonState {
	PRESSED,
	RELEASED,
	HELD
};

struct UserState {
	std::map<int, ButtonState> keyState;
	std::map<int, ButtonState> mouseState;
	int modifiers = 0;
	bool firstClick = true;
};

namespace Callbacks {
	void onKey(GLFWwindow* window, int key, int scanCode, int action, int modifiers);
	void onMouseButton(GLFWwindow* window, int button, int action, int modifiers);
	void onMouseMove(GLFWwindow* window, double x, double y);
}