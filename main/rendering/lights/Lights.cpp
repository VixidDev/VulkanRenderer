#include "Lights.hpp"

#include <fstream>
#include <algorithm>

namespace Lights {

	std::vector<Light> pointLights;
	std::vector<Light> directionalLights;
	std::vector<Light> spotLights;

	int parse(const std::string& filename) {
		// Check for correct extension
		std::size_t ext = filename.find(".vl");
		if (ext == std::string::npos) {
			std::fprintf(stderr, "Lights::parse(): Incorrect file extension passed in! Expected '.vl' found '%s'.\n", filename.substr(ext).c_str());
			return 0;
		}

		std::ifstream file(filename);

		// Check if file has badbit set
		if (file.bad()) {
			std::fprintf(stderr, "Lights::parse(): Could not read file '%s'\n", filename.c_str());
			return 0;
		}

		// Read first line to check for unique header
		std::string first;
		if (std::getline(file, first); first != "#Lights (vl)") {
			std::fprintf(stderr, "Lights::parse(): Input file has missing or incorrect file header!\n");
			return 0;
		}

		// Parse lines
		int nbDirectionalLights = 0; // Counter to issue warning if multiple directional lights are defined
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
			int shadowCasting;

			if (line.starts_with("point:")) {
				if (int res =
					std::sscanf(line.c_str(), "%*s %f %f %f %f %f %f %d %d",
						&pos.x, &pos.y, &pos.z, &colour.x, &colour.y, &colour.z, &intensity, &shadowCasting);
						res != 8) {
					std::fprintf(stderr, "Lights::parse(): Line: '%s' could not be parsed correctly! Skipping this light.\n", line.c_str());
					continue;
				}

				if (shadowCasting == 0 || shadowCasting == 1) {
					if (shadowCasting == 1) shadowCasting = true;
					if (shadowCasting == 0) shadowCasting = false;
					pointLights.emplace_back(Light(LightType::POINT, pos, glm::vec3(0.0f), colour, intensity, 0.0f, 0.0f, shadowCasting));
				} else {
					std::fprintf(stderr, "Lights::parse(): Line: '%s' could not be parsed correctly! Skipping this light.\n", line.c_str());
					continue;
				}
			} else if (line.starts_with("directional:")) {
				if (int res =
					std::sscanf(line.c_str(), "%*s %f %f %f %f %f %f %d %d",
						&direction.x, &direction.y, &direction.z, &colour.x, &colour.y, &colour.z, &intensity, &shadowCasting);
						res != 8) {
					std::fprintf(stderr, "Lights::parse(): Line: '%s' could not be parsed correctly! Skipping this light.\n", line.c_str());
					continue;
				}

				if (shadowCasting == 0 || shadowCasting == 1) {
					if (shadowCasting == 1) shadowCasting = true;
					if (shadowCasting == 0) shadowCasting = false;
					directionalLights.emplace_back(Light(LightType::DIRECTIONAL, glm::vec3(0.0f), direction, colour, intensity, 0.0f, 0.0f, shadowCasting));
				} else {
					std::fprintf(stderr, "Lights::parse(): Line: '%s' could not be parsed correctly! Skipping this light.\n", line.c_str());
					continue;
				}

				nbDirectionalLights++;
			} else if (line.starts_with("spot:")) {
				if (int res =
					std::sscanf(line.c_str(), "%*s %f %f %f %f %f %f %f %f %f %d %f %f %d",
						&pos.x, &pos.y, &pos.z, &direction.x, &direction.y, &direction.z, &colour.x, &colour.y, &colour.z, &intensity, &innerAngle, &outerAngle, &shadowCasting);
						res != 13) {
					std::fprintf(stderr, "Lights::parse(): Line: '%s' could not be parsed correctly! Skipping this light.\n", line.c_str());
					continue;
				}

				if (shadowCasting == 0 || shadowCasting == 1) {
					if (shadowCasting == 1) shadowCasting = true;
					if (shadowCasting == 0) shadowCasting = false;
					spotLights.emplace_back(Light(LightType::SPOT, pos, direction, colour, intensity, innerAngle, outerAngle, shadowCasting));
				} else {
					std::fprintf(stderr, "Lights::parse(): Line: '%s' could not be parsed correctly! Skipping this light.\n", line.c_str());
					continue;
				}
			}
		}

		file.close();

		if (nbDirectionalLights > 1)
			std::fprintf(stderr, "Lights::parse(): Multiple directional lights parsed! Having more than one directional light is undefined and will most likely break lighting!\n");

		return 1;
	}

	std::vector<Light>& getPointLights() {
		return pointLights;
	}

	std::vector<Light>& getDirectionalLights() {
		return directionalLights;
	}

	std::vector<Light>& getSpotLights() {
		return spotLights;
	}

	constexpr std::size_t getNbPointLights() {
		return pointLights.size();
	}

	constexpr std::size_t getNbDirectionalLights() {
		return directionalLights.size();
	}

	constexpr std::size_t getNbSpotLights() {
		return spotLights.size();
	}

	constexpr std::size_t getNbShadowPointLights() {
		return std::ranges::count_if(pointLights, [](const Light& light) {
			return light.isShadowCasting();
		});
	}

	constexpr std::size_t getNbShadowDirectionalLights() {
		return std::ranges::count_if(directionalLights, [](const Light& light) {
			return light.isShadowCasting();
		});
	}

	constexpr std::size_t getNbShadowSpotLights() {
		return std::ranges::count_if(spotLights, [](const Light& light) {
			return light.isShadowCasting();
		});
	}

}