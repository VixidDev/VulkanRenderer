#include "LightsParser.hpp"

#include <fstream>

#include "PointLight.hpp"
#include "DirectionalLight.hpp"
#include "SpotLight.hpp"

namespace LightsParser {

	int parseLights(const std::string& filename, std::vector<_Light>& lightsOut) {
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
			char type;
			glm::vec3 pos;
			glm::vec3 lookAt;

			if (line.starts_with("point:")) {
				if (int res = 
					std::sscanf(line.c_str(), "%*s %c %f %f %f", &type, &pos.x, &pos.y, &pos.z); 
					res != 4) 
				{
					std::fprintf(stderr, "parseLights(): Line: '%s' could not be parsed correctly! Skipping this light.\n", line.c_str());
					continue;
				}

				lightsOut.emplace_back(std::make_unique<PointLight>(pos, (type == 'd')));
			} else if (line.starts_with("directional:")) {
				if (int res = 
					std::sscanf(line.c_str(), "%*s %c %f %f %f %f %f %f", &type, &pos.x, &pos.y, &pos.z, &lookAt.x, &lookAt.y, &lookAt.z); 
					res != 7) 
				{
					std::fprintf(stderr, "parseLights(): Line: '%s' could not be parsed correctly! Skipping this light.\n", line.c_str());
					continue;
				}

				lightsOut.emplace_back(std::make_unique<DirectionalLight>(pos, lookAt, (type == 'd')));
			} else if (line.starts_with("spot:")) {
				if (int res =
					std::sscanf(line.c_str(), "%*s %c %f %f %f %f %f %f", &type, &pos.x, &pos.y, &pos.z, &lookAt.x, &lookAt.y, &lookAt.z);
					res != 7) 
				{
					std::fprintf(stderr, "parseLights(): Line: '%s' could not be parsed correctly! Skipping this light.\n", line.c_str());
					continue;
				}

				lightsOut.emplace_back(std::make_unique<SpotLight>(pos, lookAt, (type == 'd')));
			}
		}

		file.close();

		return 1;
	}

}