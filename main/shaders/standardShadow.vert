#version 450

layout(location = 0) in vec3 iPosition;
layout(location = 1) in vec2 iTexCoord;

layout(location = 0) out vec2 v2fTexCoord;

layout(push_constant) uniform PushConstant {
	mat4 depthMatrix;
} pConsts;

void main() {
	v2fTexCoord = iTexCoord;

	gl_Position = pConsts.depthMatrix * vec4(iPosition, 1.0f);
}