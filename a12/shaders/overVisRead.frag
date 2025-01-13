#version 450

layout(input_attachment_index = 0, binding = 0) uniform isubpassInput inputDepth;

layout(location = 0) out vec4 oColor;

void main() {
	ivec4 stencilVal = subpassLoad(inputDepth);

	oColor = vec4(stencilVal.r / 255.0f);
}