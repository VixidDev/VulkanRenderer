#version 450

layout(set = 0, binding = 0) uniform MVP {
	mat4 projection;
	mat4 view;
	vec4 camPos;
} mvp;

layout(location = 0) out vec4 oColour;

void main() {
	oColour = vec4(vec3(0.05), 0.0);
}