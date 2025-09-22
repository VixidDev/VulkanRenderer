#version 450

layout(location = 0) in vec3 iPosition;

layout(set = 0, binding = 0) uniform MVP {
	mat4 projection;
	mat4 view;
	vec4 camPos;
} mvp;

void main() {
	gl_Position = mvp.projection * mvp.view * vec4(iPosition, 1.0);
}