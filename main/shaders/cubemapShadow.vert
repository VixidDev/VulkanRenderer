#version 450

layout(location = 0) in vec3 iPosition;

layout(set = 0, binding = 0) uniform UScene {
	mat4 depthMVP;
} uScene;

layout(location = 0) out vec3 worldPos;

void main() {
	vec4 pos = uScene.depthMVP * vec4(iPosition, 1.0f);
	worldPos = pos.xyz;
	gl_Position = pos;
}