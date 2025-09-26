#include <cstdio>
#include <exception>

#include "Driver.hpp"

int main() try {
	Driver driver = Driver();

	if (!driver.init()) {
		std::fprintf(stderr, "Failed to initialise driver!\n");
		return 1;
	}

	driver.run();

	return 0;
} catch (const std::exception& error) {
	std::fprintf(stderr, "\nError: %s\n", error.what());
	return 1;
}