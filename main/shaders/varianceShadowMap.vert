#version 450

layout(location = 0) in vec3 iPosition;
layout(location = 1) in vec2 iTexCoord;

layout(location = 0) out float v2fLightDepth;
layout(location = 1) out vec2 v2fTexCoord;

layout(push_constant) uniform PushConstants {
	mat4 lightViewProj;
	mat4 lightView;
} pConsts;

void main() {
	vec4 lightViewPos = pConsts.lightView * vec4(iPosition, 1.0);

	v2fLightDepth = lightViewPos.z;
	v2fTexCoord = iTexCoord;

	gl_Position = pConsts.lightViewProj * vec4(iPosition, 1.0);
}
