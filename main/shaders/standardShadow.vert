#version 450

layout(location = 0) in vec3 iPosition;

layout(push_constant) uniform PushConstant {
	mat4 depthMatrix;
} pConsts;

void main() {
	gl_Position = pConsts.depthMatrix * vec4(iPosition, 1.0f);
}