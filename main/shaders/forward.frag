#version 450

#include "lighting.glsl"

layout(location = 0) in vec3 v2fPosition;
layout(location = 1) in vec2 v2fTexCoord;
layout(location = 2) in vec4 v2fFallbackNormal;
layout(location = 3) in mat3 v2fTBN;

layout(set = 0, binding = 0) uniform MVP {
	mat4 projection;
	mat4 view;
	vec4 camPos;
} mvp;

layout(set = 1, binding = 0) readonly buffer Lights {
	ShaderLight lights[];
};

layout(set = 2, binding = 0) uniform sampler2D uSSAO;

layout(set = 3, binding = 0) uniform sampler2D uTexColour;
layout(set = 3, binding = 1) uniform sampler2D uMetalness;
layout(set = 3, binding = 2) uniform sampler2D uRoughness;
layout(set = 3, binding = 3) uniform sampler2D uAlphaMask;
layout(set = 3, binding = 4) uniform sampler2D uNormalMap;
layout(set = 3, binding = 5) uniform sampler2D uEmissive;

layout(push_constant) uniform PushConstants {
	float emissiveStrength;
	float brightnessThreshold;
	float shadowBias;
	float bleedReduction;
	int ssaoEnabled;
	float ssaoExp;
} pConsts;

layout(location = 0) out vec4 oColour;
layout(location = 1) out vec4 oBrightness;

layout(constant_id = 0) const int NUM_LIGHTS = 0;

void main() {
	// Discard fragments that fail alpha test
	float alphaValue = texture(uAlphaMask, v2fTexCoord).a;
	if (alphaValue < 0.5) discard;

	// Get normal from either fallback or normal map
	vec3 normal;
	// w component will be 1 if TBN contains NaNs
	if (v2fFallbackNormal.w == 1.0) {
		normal = normalize(v2fFallbackNormal.xyz);
	} else {
		vec3 tangentNormal = texture(uNormalMap, v2fTexCoord).rgb;
		tangentNormal = tangentNormal * 2.0 - 1.0;
		normal = normalize(v2fTBN * tangentNormal);
	}

	vec3 viewDir = normalize(mvp.camPos.xyz - v2fPosition);
	float nDotV  = dot(normal, viewDir); 
	vec3 albedo     = texture(uTexColour, v2fTexCoord).rgb;
	float metalness = texture(uMetalness, v2fTexCoord).r;
	float roughness = texture(uRoughness, v2fTexCoord).r;
	float a = roughness * roughness;
	float a2 = a * a;

	vec3 F0 = vec3(0.04);
	F0 = mix(F0, albedo, metalness);

	vec3 Lo = vec3(0.0);
	// Iterate over all lights
	for (int i = 0; i < NUM_LIGHTS; i++) {
		ShaderLight light = lights[i];

		vec3 lightPos = light.positionAndLightType.xyz;
		float distToLight = length(lightPos - v2fPosition);
		vec3 lightDir = normalize(lightPos - v2fPosition);

		float attenuation = 1.0;
		if (light.positionAndLightType.w == 1) {
			// Directional lights have an attenuation of 1 so keep as is.
			// Light dir should be parallel for every fragment for directional lights
			lightDir = -light.directionAndMapIndex.xyz;
		} else {
			// Keep point and spot lights with squared attenuation
			attenuation = 1 / (distToLight * distToLight);
		}

		vec3 lightColour = light.colourAndIntensity.rgb;
		float intensity  = light.colourAndIntensity.w;
		vec3 radiance    = lightColour * intensity * attenuation;

		vec3 brdf = CookTorranceBRDF(lightDir, viewDir, normal, nDotV, metalness, roughness, 
									 a, a2, F0, albedo, radiance, 1.0);

		Lo += brdf;
	}

	vec2 screenSize = textureSize(uSSAO, 0);
	vec2 screenSpaceUV = gl_FragCoord.xy / screenSize;
	float ssao = pConsts.ssaoEnabled == 1 ? texture(uSSAO, screenSpaceUV).r : 1.0;
	ssao = pow(ssao, pConsts.ssaoExp);

	// Add ambient aspect and account for AO
	vec3 ambient = vec3(0.03) * albedo * ssao;
	vec3 colour = ambient + Lo;

	// Add any emissive colour
	vec3 emissive = texture(uEmissive, v2fTexCoord).rgb;
	colour += emissive * pConsts.emissiveStrength;

	oColour = vec4(colour, 1.0);

	// Write any fragments that pass the threshold to the brightness texture
	// for bloom post process effect
	float brightness = dot(oColour.rgb, vec3(0.2126, 0.7152, 0.0722));
	if (brightness > pConsts.brightnessThreshold) {
		oBrightness = oColour;
	} else {
		oBrightness = vec4(0.0, 0.0, 0.0, 1.0);
	}
}
