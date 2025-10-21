#version 450

layout(location = 0) in float v2fLightDepth;
layout(location = 1) in vec2 v2fTexCoord;

layout(set = 0, binding = 0) uniform ClipPlanes {
	float far;
	float near;
} planes;

layout(set = 1, binding = 0) uniform sampler2D uAlphaMask;

layout(location = 0) out vec2 oMoments;

void main() {
	float alpha = texture(uAlphaMask, v2fTexCoord).a;
	if (alpha < 0.5) discard;

	float linearDepth = (-v2fLightDepth - planes.near) / (planes.far - planes.near);

	oMoments = vec2(linearDepth, linearDepth * linearDepth);
}