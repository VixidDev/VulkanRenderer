#version 450

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

layout(location = 0) out vec4 gBuffer1; // normals = rgb, metalness = a
layout(location = 1) out vec4 gBuffer2; // albedo = rgb, roughness = a

void main() {
	vec3 normal;
	if (v2fFallbackNormal.w == 1.0) {
		normal = v2fFallbackNormal.xyz;
	} else {
		normal = v2fTBN * normalize(texture(uNormalMap, v2fTexCoord).rgb * 2.0 - 1.0);
	}

	gBuffer1.rgb = normal;
	gBuffer1.a = texture(uMetalness, v2fTexCoord).r;

	// Discard fragments that fail alpha test
	float alphaValue = texture(uAlphaMask, v2fTexCoord).a;
	if (alphaValue < 0.5) discard;

	gBuffer2.rgb = texture(uTexColour, v2fTexCoord).rgb;
	gBuffer2.a = texture(uRoughness, v2fTexCoord).r;
}