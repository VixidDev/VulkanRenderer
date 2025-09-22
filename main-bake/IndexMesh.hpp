#pragma once

#include <vector>

#include <cstdint>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>


struct TriangleSoup {
	std::vector<glm::vec3> vert;
	std::vector<glm::vec3> norm;
	std::vector<glm::vec2> text;
};

struct IndexedMesh {
	std::vector<glm::vec3> vert;
	std::vector<glm::vec3> norm;
	std::vector<glm::vec2> text;

	std::vector<glm::vec4> tangent;
	std::vector<std::uint32_t> tangentComp;

	std::vector<std::uint32_t> indices;

	glm::vec3 aabbMin, aabbMax;

	IndexedMesh();
};

IndexedMesh makeIndexedMesh(const TriangleSoup& triSoup, float errorTol = 1e-6f, int i = 0);

void ensureNormals(IndexedMesh& iMesh);