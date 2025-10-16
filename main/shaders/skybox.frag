#version 450

layout(location = 0) in vec3 v2fPosition;

layout(set = 1, binding = 0) uniform samplerCube skyboxTexture;

layout(location = 0) out vec4 oColour;

void main() {
	oColour = texture(skyboxTexture, v2fPosition);
}