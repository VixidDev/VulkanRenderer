#version 450

//layout(input_attachment_index = 0, binding = 0) uniform subpassInput inputColor;
layout(input_attachment_index = 0, binding = 0) uniform isubpassInput inputDepth;

layout(location = 0) out vec4 oColor;

void main() {
	ivec4 d = subpassLoad(inputDepth);
	oColor = vec4(d.r / 255.0f);
}