#version 450

#include "lighting.glsl"

layout(location = 0) in vec2 v2fTexCoord;

layout(set = 0, binding = 0) uniform sampler2D gBuffer1;  // rgb: normals
layout(set = 0, binding = 1) uniform sampler2D gBuffer2;  // rgb: albedo,   a = roughness
layout(set = 0, binding = 2) uniform sampler2D gBuffer3;  // rgb: emissive, a = metalness
layout(set = 0, binding = 3) uniform sampler2D uDepth;

layout(set = 1, binding = 0) uniform MVP {
	mat4 projection;
	mat4 view;
	vec4 camPos;
} mvp;

layout(set = 2, binding = 0) readonly buffer Lights {
	ShaderLight lights[];
};

layout(set = 3, binding = 0) uniform InverseMatrices {
	mat4 invViewProj;
	mat4 invProj;
	mat4 invView;
} inverses;

layout(set = 4, binding = 0) uniform sampler2D uSSAO;

layout(push_constant) uniform PushConstants {
	int lightCount;
	float emissiveStrength;
	float brightnessThreshold;
	float shadowBias;
	int ssaoEnabled;
	float ssaoExp;
} pConsts;

layout(location = 0) out vec4 oColour;
layout(location = 1) out vec4 oBrightness;

layout(constant_id = 0) const int VIEW_SPACE_NORMALS = 0;

vec3 posFromDepth(float depth) {
	vec4 clipSpace = vec4(v2fTexCoord * 2.0 - 1.0, depth, 1.0);
	vec4 viewSpace = inverses.invViewProj * clipSpace;
	vec3 worldSpace = viewSpace.xyz / viewSpace.w;
	return worldSpace;
}

void main() {
    float depth = texture(uDepth, v2fTexCoord).r;
	
    // Fragments with depth of 1 are fragments that weren't drawn
    // to in the previous pass, without this, the 'skybox' would be
    // black instead of the clear color
    if (depth == 1.0) discard;
    
    // Get world space vertex position from depth buffer
    vec3 pos = posFromDepth(depth);

	vec3 normal = texture(gBuffer1, v2fTexCoord).rgb;
	// Map normals from [0, 1] (gBuffer format is UNORM) back to [-1, 1]
	normal = normal * 2.0 - 1.0;

	if (VIEW_SPACE_NORMALS == 1.0) {
		normal = normalize(mat3(inverses.invView) * normal);
	}

	vec3 viewDir = normalize(mvp.camPos.xyz - pos);

	vec3 albedo     = texture(gBuffer2, v2fTexCoord).rgb;
	float metalness = texture(gBuffer3, v2fTexCoord).a;
	float roughness = texture(gBuffer2, v2fTexCoord).a;

	vec3 F0 = vec3(0.04);
	F0 = mix(F0, albedo, metalness);

	vec3 Lo = vec3(0.0);
    // Iterate over all lights
    for (int i = 0; i < pConsts.lightCount; i++) {

		vec3 lightPos = lights[i].positionAndLightType.xyz;
		float distToLight = length(lightPos - pos);
		vec3 lightDir = normalize(lightPos - pos);
        
		float attenuation = 1.0;
		if (lights[i].positionAndLightType.w == 1) {
			// Directional lights have an attenuation of 1 so keep as is.
			// Light dir should be parallel for every fragment for directional lights
			lightDir = -lights[i].directionAndMapIndex.xyz;
		} else {
			// Keep point and spot lights with squared attenuation
			attenuation = 1 / (distToLight * distToLight);
		}

		vec3 lightColour = lights[i].colourAndIntensity.rgb;
		float intensity  = lights[i].colourAndIntensity.w;
		vec3 radiance    = lightColour * intensity * attenuation;

		vec3 brdf = CookTorranceBRDF(lightDir, viewDir, normal, metalness, roughness, F0, albedo, radiance, 1.0);

		Lo += brdf;
	}

	// Tex coords from deferred vertex shader are already in clip space
	float ssao = pConsts.ssaoEnabled == 1 ? texture(uSSAO, v2fTexCoord).r : 1.0;
	ssao = pow(ssao, pConsts.ssaoExp);

	// Add ambient aspect and account for AO
	vec3 ambient = vec3(0.03) * albedo * ssao;
	vec3 colour = ambient + Lo;

	// Add any emissive colour
	vec3 emissive = texture(gBuffer3, v2fTexCoord).rgb;
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