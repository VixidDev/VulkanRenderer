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
		int directionalLights = 0; // Counter to issue warning if multiple directional lights are defined
		std::string line;
		while (std::getline(file, line)) {
			// Skip comments
			if (line.starts_with("//")) continue;

			glm::vec3 pos;
			glm::vec3 direction;
			glm::vec3 colour;
			int intensity;
			float innerAngle;
			float outerAngle;

			if (line.starts_with("point:")) {
				if (int res =
					std::sscanf(line.c_str(), "%*s %f %f %f %f %f %f %d", &pos.x, &pos.y, &pos.z, &colour.x, &colour.y, &colour.z, &intensity);
					res != 7) 
				{
					std::fprintf(stderr, "parseLights(): Line: '%s' could not be parsed correctly! Skipping this light.\n", line.c_str());
					continue;
				}
					lightsOut.emplace_back(Light(LightType::POINT, pos, glm::vec3(0.0f), colour, intensity));
			} else if (line.starts_with("directional:")) {
				int temp = 0;
				if (int res =
					std::sscanf(line.c_str(), "%*s %f %f %f %f %f %f %d", &direction.x, &direction.y, &direction.z, &colour.x, &colour.y, &colour.z, &intensity);
					res != 7) 
				{
					std::fprintf(stderr, "parseLights(): Line: '%s' could not be parsed correctly! Skipping this light.\n", line.c_str());
					continue;
				}
					lightsOut.emplace_back(Light(LightType::DIRECTIONAL, glm::vec3(0.0f), direction, colour, intensity));
					directionalLights++;
			} else if (line.starts_with("spot:")) {
				if (int res =
					std::sscanf(line.c_str(), "%*s %f %f %f %f %f %f %f %f %f %d %f %f", &pos.x, &pos.y, &pos.z, &direction.x, &direction.y, &direction.z, &colour.x, &colour.y, &colour.z, &intensity, &innerAngle, &outerAngle);
					res != 12) 
				{
					std::fprintf(stderr, "parseLights(): Line: '%s' could not be parsed correctly! Skipping this light.\n", line.c_str());
					continue;
				}
					lightsOut.emplace_back(Light(LightType::SPOT, pos, direction, colour, intensity, innerAngle, outerAngle));
			}
		}

		file.close();

		if (directionalLights > 1)
			std::fprintf(stderr, "parseLights(): Multiple directional lights parsed! Having more than one directional light is undefined and will most likely break lighting!\n");

		return 1;
	}

}