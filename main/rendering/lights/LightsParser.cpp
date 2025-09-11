#include "LightsParser.hpp"

#include <fstream>

namespace LightsParser {

	int parseLights(const std::string& filename, std::vector<Light>& lightsOut) {
		// Check for correct extension
		std::size_t ext = filename.find(".vl");
		if (ext == std::string::npos) {
			std::fprintf(stderr, "parseLights(): Incorrect file extension passed in! Expected '.vl' found '%s'.\n", filename.substr(ext).c_str());
			return 0;
		}

		std::ifstream file(filename);

		// Check if file has badbit set
		if (file.bad()) {
			std::fprintf(stderr, "parseLights(): Could not read file '%s'\n", filename.c_str());
			return 0;
		}

		// Read first line to check for unique header
		std::string first;
		if (std::getline(file, first); first != "#Lights (vl)") {
			std::fprintf(stderr, "parseLights(): Input file has missing or incorrect file header!\n");
			return 0;
		}

		// Parse lines
		std::string line;
		while (std::getline(file, line)) {
			glm::vec3 pos;
			glm::vec3 lookAt;

			if (line.starts_with("point:")) {
				if (int res = 
					std::sscanf(line.c_str(), "%*s %f %f %f", &pos.x, &pos.y, &pos.z); 
					res != 3) 
				{
					std::fprintf(stderr, "parseLights(): Line: '%s' could not be parsed correctly! Skipping this light.\n", line.c_str());
					continue;
				}

				lightsOut.emplace_back(Light(LightType::POINT, pos, glm::vec3(0.0f)));
			} else if (line.starts_with("directional:")) {
				if (int res = 
					std::sscanf(line.c_str(), "%*s %f %f %f %f %f %f", &pos.x, &pos.y, &pos.z, &lookAt.x, &lookAt.y, &lookAt.z); 
					res != 6) 
				{
					std::fprintf(stderr, "parseLights(): Line: '%s' could not be parsed correctly! Skipping this light.\n", line.c_str());
					continue;
				}

				lightsOut.emplace_back(Light(LightType::DIRECTIONAL, pos, lookAt));
			} else if (line.starts_with("spot:")) {
				if (int res =
					std::sscanf(line.c_str(), "%*s %f %f %f %f %f %f", &pos.x, &pos.y, &pos.z, &lookAt.x, &lookAt.y, &lookAt.z);
					res != 6) 
				{
					std::fprintf(stderr, "parseLights(): Line: '%s' could not be parsed correctly! Skipping this light.\n", line.c_str());
					continue;
				}

				lightsOut.emplace_back(Light(LightType::SPOT, pos, lookAt));
			}
		}

		file.close();

		return 1;
	}

}