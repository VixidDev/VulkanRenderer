namespace vk {
	// UniqueHandle<> implementation
	template<typename tHandle, typename tParent, DestroyFn<tParent, tHandle>& tDestroyFn>
	inline
		UniqueHandle<tHandle, tParent, tDestroyFn>::UniqueHandle(tParent parent, tHandle handle) noexcept
		: handle(handle)
		, mParent(parent) {}

	template<typename tHandle, typename tParent, DestroyFn<tParent, tHandle>& tDestroyFn>
	inline
		UniqueHandle<tHandle, tParent, tDestroyFn>::~UniqueHandle() {
		if (VK_NULL_HANDLE != handle) {
			assert(VK_NULL_HANDLE != mParent);
			tDestroyFn(mParent, handle, nullptr);
		}
	}

	template<typename tHandle, typename tParent, DestroyFn<tParent, tHandle>& tDestroyFn>
	inline UniqueHandle<tHandle, tParent, tDestroyFn>::UniqueHandle(UniqueHandle&& other) noexcept
		: handle(std::exchange(other.handle, VK_NULL_HANDLE))
		, mParent(std::exchange(other.mParent, VK_NULL_HANDLE)) {}

	template<typename tHandle, typename tParent, DestroyFn<tParent, tHandle>& tDestroyFn>
	inline UniqueHandle<tHandle, tParent, tDestroyFn>& UniqueHandle<tHandle, tParent, tDestroyFn>::operator=(UniqueHandle&& other) noexcept {
		std::swap(handle, other.handle);
		std::swap(mParent, other.mParent);
		return *this;
	}
}