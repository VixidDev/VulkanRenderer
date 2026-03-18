#pragma once

#include "Textures.hpp"
#include "../../impl/Textures.hpp"

#include <optional>

enum AttachmentLoadOp {
	CLEAR = VK_ATTACHMENT_LOAD_OP_CLEAR,
	LOAD = VK_ATTACHMENT_LOAD_OP_LOAD,
	DONT_CARE = VK_ATTACHMENT_LOAD_OP_DONT_CARE
};

enum AttachmentStoreOp {
	STORE = VK_ATTACHMENT_STORE_OP_STORE,
	DONT_CARE = VK_ATTACHMENT_STORE_OP_DONT_CARE
};

struct AttachmentDesc {
	Texture texture;
	AttachmentLoadOp loadOp;
	AttachmentStoreOp storeOp;
	std::optional<ImageLayout> finalLayout = std::nullopt;
	std::optional<ImageLayout> initialLayout = std::nullopt;
};