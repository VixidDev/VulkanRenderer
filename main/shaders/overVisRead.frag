#version 450

layout(input_attachment_index = 0, binding = 0) uniform isubpassInput inputDepth;

layout(location = 0) out vec4 oColor;

void main() {
	ivec4 stencilVal = subpassLoad(inputDepth);

	vec4 g = vec4(0.0f, 0.3f, 0.0f, 0.1f);

	oColor = g + vec4(vec3(stencilVal.r / 25.0f), 0.0f);
}