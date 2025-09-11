#version 450

layout(location = 0) in vec3 iPosition;

layout(push_constant) uniform PushConstant {
	mat4 depthMVP;
} pConsts;

layout(location = 0) out vec3 worldPos;

void main() {
	worldPos = iPosition;
	gl_Position = pConsts.depthMVP * vec4(iPosition, 1.0f);
}