#version 450

layout(location = 0) in vec3 iPosition;

layout(set = 0, binding = 0) uniform UScene {
	mat4 depthMVP;
} uScene;

layout(location = 0) out vec3 worldPos;

void main() {
	worldPos = iPosition;
	gl_Position = uScene.depthMVP * vec4(iPosition, 1.0f);
}