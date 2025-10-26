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

layout(set = 5, binding = 0) uniform samplerCubeArray pointLightShadows;
layout(set = 5, binding = 1) uniform sampler2D sunShadow;
layout(set = 5, binding = 2) uniform sampler2DArray spotLightShadows;

layout(set = 6, binding = 0) uniform ClipPlanes {
	float far;
	float near;
} planes;

layout(set = 7, binding = 0) readonly buffer LightSpaceMatrices {
	mat4 lightSpaceMatrices[];
};

layout(push_constant) uniform PushConstants {
	int lightCount;
	float emissiveStrength;
	float brightnessThreshold;
	float shadowBias;
	float bleedReduction;
	int ssaoEnabled;
	float ssaoExp;
} pConsts;

layout(location = 0) out vec4 oColour;
layout(location = 1) out vec4 oBrightness;

layout(constant_id = 0) const int VIEW_SPACE_NORMALS = 0;

const mat4 biasMat = mat4( 
	0.5, 0.0, 0.0, 0.0,
	0.0, 0.5, 0.0, 0.0,
	0.0, 0.0, 1.0, 0.0,
	0.5, 0.5, 0.0, 1.0);

vec3 posFromDepth(float depth) {
	vec4 clipSpace = vec4(v2fTexCoord * 2.0 - 1.0, depth, 1.0);
	vec4 viewSpace = inverses.invViewProj * clipSpace;
	vec3 worldSpace = viewSpace.xyz / viewSpace.w;
	return worldSpace;
}

float ChebyshevUpperBound(vec2 moments, float depth) {
	float tDepth = depth - pConsts.shadowBias;

	if (tDepth <= moments.x) {
		return 1.0;
	}

	float variance = moments.y - (moments.x * moments.x);
	variance = max(variance, 0.00002);

	float diff = tDepth - moments.x;
	float diff2 = diff * diff;

	float pMax = variance / (variance + diff2);
	pMax = clamp((pMax - pConsts.bleedReduction) / (1.0 - pConsts.bleedReduction), 0.0, 1.0);

	return (tDepth <= moments.x) ? 1.0 : pMax;
}

float calculateShadow(ShaderLight light, vec3 pos) {
	float shadow = 0.0;

	int lightType	   = int(light.positionAndLightType.w);
	int shadowMapIndex = int(light.directionAndMapIndex.w);
	int lightSpaceMatrixIndex = int(light.extra.z);

	// Directional / Spot light related vars
	mat4 lightSpaceMatrix = biasMat * lightSpaceMatrices[lightSpaceMatrixIndex];
	vec4 lightSpacePos = lightSpaceMatrix * vec4(pos, 1.0);
	vec3 shadowCoord   = lightSpacePos.xyz / lightSpacePos.w;

	vec2 moments;
	float fragDepth;
	vec3 lightToFrag;

	switch (lightType) {
	case 0: // Point light
		lightToFrag = pos - light.positionAndLightType.xyz;

		fragDepth = length(lightToFrag) / planes.far;
		vec3 dir = normalize(lightToFrag);

		moments = texture(pointLightShadows, vec4(dir, shadowMapIndex)).rg;

		shadow = ChebyshevUpperBound(moments, fragDepth);
		break;
	case 1: // Directional light
		// Scaled shadow bias based on distance from camera.
		// Will result in slightly different shadowing from
		// that in forward rendering, especially at far distances.
		float distanceToCamera = length(pos - mvp.camPos.xyz);
		float bias = distanceToCamera > 100.0 ? 0.00005 : 0.00001;
		shadowCoord.z -= distanceToCamera * bias;

		moments = texture(sunShadow, shadowCoord.xy).rg;
		fragDepth = shadowCoord.z;

		shadow = ChebyshevUpperBound(moments, fragDepth);
		break;
	case 2: // Spot light
		lightToFrag = pos - light.positionAndLightType.xyz;
		fragDepth = length(lightToFrag) / planes.far;

		moments = texture(spotLightShadows, vec3(shadowCoord.xy, shadowMapIndex)).rg;

		shadow = ChebyshevUpperBound(moments, fragDepth);
		break;
	}

	return shadow;
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

	if (VIEW_SPACE_NORMALS == 1) {
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

		float shadow = 1.0;
		// If light is a shadow caster, calculate shadow
		if (lights[i].extra.w == 1) {
			shadow = calculateShadow(lights[i], pos);
		}

		vec3 brdf = CookTorranceBRDF(lightDir, viewDir, normal, metalness, roughness, F0, albedo, radiance, shadow);

		if (lights[i].positionAndLightType.w == 2) {
			vec3 lightToFrag = normalize(pos - lights[i].positionAndLightType.xyz);
			float theta = dot(lightToFrag, lights[i].directionAndMapIndex.xyz);
			float innerConeAngle = lights[i].extra.x;
			float outerConeAngle = lights[i].extra.y;
			float intensity = (theta - outerConeAngle) / (innerConeAngle - outerConeAngle);
			brdf = smoothstep(0.0, 1.0, intensity) * brdf;
		}

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