#pragma once

class ITextureBufferListener {
public:
	virtual void onTextureBufferRecreated() = 0;
	virtual ~ITextureBufferListener() = default;
};