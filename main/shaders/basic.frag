#version 450

#define PI 3.14159265359

layout(location = 0) in vec3 v2fPosition;
layout(location = 1) in vec2 v2fTexCoord;
layout(location = 2) in vec4 v2fFallbackNormal;
layout(location = 3) in mat3 v2fTBN;

layout(set = 0, binding = 0) uniform MVP {
	mat4 projection;
	mat4 view;
	vec4 camPos;
} mvp;

layout(set = 1, binding = 0) uniform sampler2D uTexColour;
layout(set = 1, binding = 1) uniform sampler2D uMetalness;
layout(set = 1, binding = 2) uniform sampler2D uRoughness;
layout(set = 1, binding = 3) uniform sampler2D uAlphaMask;
layout(set = 1, binding = 4) uniform sampler2D uNormalMap;
layout(set = 1, binding = 5) uniform sampler2D uEmissive;

layout(location = 0) out vec4 oColour;

void main() {
	// Discard fragments that fail alpha test
	float alphaValue = texture(uAlphaMask, v2fTexCoord).a;
	if (alphaValue < 0.5) discard;

	vec3 normal;
	if (v2fFallbackNormal.w == 1.0) {
		normal = normalize(v2fFallbackNormal.xyz);
	} else {
		vec3 tangentNormal = texture(uNormalMap, v2fTexCoord).rgb;
		tangentNormal = tangentNormal * 2.0 - 1.0;
		normal = normalize(v2fTBN * tangentNormal);
	}

	float diffuse = max(dot(normal, vec3(0.34815531 -0.8703882 -0.34815531)), 0.0);
	vec3 base = texture(uTexColour, v2fTexCoord).rgb;
	
	oColour = vec4(base * (0.2 + diffuse), 1.0);
}