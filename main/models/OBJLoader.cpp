#define TINYOBJLOADER_IMPLEMENTATION

#include "OBJLoader.hpp"

namespace OBJLoader {

	int loadFromFile(const std::string& filename, OBJModel& modelOut) {
		tinyobj::ObjReader reader;

		if (!reader.ParseFromFile(filename)) {
			if (!reader.Error().empty()) {
				std::fprintf(stderr, "OBJLoader::loadFromFile(): File '%s' encountered error: '%s'\n", filename.c_str(), reader.Error().c_str());
			}
			return 0;
		}

		if (!reader.Warning().empty()) {
			std::fprintf(stdout, "OBJLoader::loadFromFile(): File '%s' encountered warning: '%s'\n", filename.c_str(), reader.Warning().c_str());
		}
		
		const tinyobj::attrib_t& attrib = reader.GetAttrib();
		const std::vector<tinyobj::shape_t>& shapes = reader.GetShapes();

		OBJModel model{};

		// Loop over shapes
		for (size_t i = 0; i < shapes.size(); i++) {
			size_t indexOffset = 0;

			// Check if we have a name yet
			if (model.name.empty()) model.name = reader.GetShapes()[i].name;

			// Loop over faces
			for (size_t f = 0; f < shapes[i].mesh.num_face_vertices.size(); f++) {
				size_t fv = size_t(shapes[i].mesh.num_face_vertices[f]);

				// Loop over vertices
				glm::vec3 vertices;
				size_t index;
				for (size_t v = 0; v < fv; v++) {
					index = shapes[i].mesh.indices[indexOffset + v].vertex_index;
					vertices[0] = attrib.vertices[3 * index + 0];
					vertices[1] = attrib.vertices[3 * index + 1];
					vertices[2] = attrib.vertices[3 * index + 2];

					model.vertices.emplace_back(vertices);
					model.indices.emplace_back(index);
				}

				indexOffset += fv;
			}
		}

		return 1;
	}

}
