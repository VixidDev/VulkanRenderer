#version 450

layout(location = 0) in vec2 v2fTexCoord;
layout(location = 1) in vec3 v2fNormal;
layout(location = 2) in mat3 v2fTBN;

layout(set = 1, binding = 0) uniform sampler2D uTexColor;
layout(set = 1, binding = 1) uniform sampler2D uMetalness;
layout(set = 1, binding = 2) uniform sampler2D uRoughness;
layout(set = 1, binding = 3) uniform sampler2D uAlphaMask;
layout(set = 1, binding = 4) uniform sampler2D uNormalMap;

layout(location = 0) out vec4 outNormals;
layout(location = 1) out vec4 outAlbedo;

void main() {
	vec3 normal = v2fTBN * normalize(texture(uNormalMap, v2fTexCoord).rgb * 2.0f - 1.0f);

	outNormals.rgb = normal;
	outNormals.a   = texture(uMetalness, v2fTexCoord).r;

	float alphaValue = texture(uAlphaMask, v2fTexCoord).a;
    if (alphaValue < 0.5) discard;

	outAlbedo.rgb = texture(uTexColor, v2fTexCoord).rgb;
	outAlbedo.a   = texture(uRoughness, v2fTexCoord).r;
}