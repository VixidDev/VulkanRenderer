#pragma once

#include <functional>

// Simple template class to Cache an object.
// Users must remember to manually call markDirty()
// on the cache when a dependent variable used in the
// recalculation function is changed.
template<typename T>
class Cache {
public:
	Cache() = default;
	Cache(std::function<T()> recalculationFunc) : recalc(std::move(recalculationFunc)) {}

	const T& get() {
		if (this->dirty) {
			this->value = this->recalc();
			this->dirty = false;
		}

		return this->value;
	}

	void markDirty() {
		this->dirty = true;
	}

	bool isDirty() const {
		return this->dirty;
	}

private:
	std::function<T()> recalc;

	T value{};
	bool dirty = true;
};
