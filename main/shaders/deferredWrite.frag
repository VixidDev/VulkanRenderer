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
layout(set = 1, binding = 5) uniform sampler2D uEmssive;

layout(location = 0) out vec4 gBuffer1; // normals = rgb (format: A2R10G10B10_UNORM)
layout(location = 1) out vec4 gBuffer2; // albedo = rgb, roughness = a
layout(location = 2) out vec4 gBuffer3; // emissive = rgb, metalness = a

void main() {
	// Discard fragments that fail alpha test
	float alphaValue = texture(uAlphaMask, v2fTexCoord).a;
	if (alphaValue < 0.5) discard;

	vec3 normal;
	if (v2fFallbackNormal.w == 1.0) {
		normal = v2fFallbackNormal.xyz;
	} else {
		vec3 tangentNormal = texture(uNormalMap, v2fTexCoord).rgb;
		tangentNormal = tangentNormal * 2.0 - 1.0;
		normal = normalize(v2fTBN * tangentNormal);
		// Map normals from [-1, 1] to [0, 1] since gBuffer format is UNORM
		//normal = normal * 0.5 + 0.5;
	}

	gBuffer1.rgb = normal;

	gBuffer2.rgb = texture(uTexColour, v2fTexCoord).rgb;
	gBuffer2.a = texture(uRoughness, v2fTexCoord).r;

	gBuffer3.rgb = texture(uEmssive, v2fTexCoord).rgb;
	gBuffer3.a = texture(uMetalness, v2fTexCoord).r;
}