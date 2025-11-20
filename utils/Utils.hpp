#pragma once

#include "glm/glm.hpp"

namespace Utils {

	glm::vec4 row(glm::mat4& mat, int r) {
		return glm::vec4(mat[0][r], mat[1][r], mat[2][r], mat[3][r]);
	}

}