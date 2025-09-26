#include "Mouse.hpp"

namespace Mouse {

	namespace {
		float x;
		float y;
	}

	void setX(float xIn) {
		x = xIn;
	}

	void setY(float yIn) {
		y = yIn;
	}

	float getX() {
		return x;
	}

	float getY() {
		return y;
	}

}