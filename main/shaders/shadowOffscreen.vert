#version 450

layout(location = 0) in vec3 iPosition;

layout(set = 0, binding = 0) uniform UScene {
	mat4 depthMVP;
} uScene;

void main() {
	gl_Position = uScene.depthMVP * vec4(iPosition, 1.0f);
}