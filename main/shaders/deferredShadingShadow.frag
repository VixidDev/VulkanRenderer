#version 450

#define PI 3.14159265359
#define SHADOW_BIAS 0.0001

layout(location = 0) in vec2 v2fTexCoord;

layout(set = 0, input_attachment_index = 0, binding = 0) uniform subpassInput gBuffer1;  // normals = rgb
layout(set = 0, input_attachment_index = 1, binding = 1) uniform subpassInput gBuffer2;  // albedo = rgb, roughtness = a
layout(set = 0, input_attachment_index = 2, binding = 2) uniform subpassInput gBuffer3;  // emissive = rgb, metalness = a
layout(set = 0, input_attachment_index = 3, binding = 3) uniform subpassInput inputDepth;

layout(set = 1, binding = 0) uniform MVP {
	mat4 projection;
	mat4 view;
	vec4 camPos;
} mvp;

layout(set = 2, binding = 0) uniform samplerCubeArrayShadow pointLightShadows;
layout(set = 3, binding = 0) uniform sampler2DShadow sunShadow;
//layout(set = 5, binding = 0) uniform sampler2DArrayShadow spotLightShadows;

layout(set = 4, binding = 0) uniform ClipPlanes {
	float far;
	float near;
} planes;

struct ShaderLight {
	vec3 position;
	vec3 direction;
	vec3 colour;
	ivec3 metadata;
	// metadata.x = lightType // 0 - Point Light, 1 - Directional Light, 2 - Spot light
	// metadata.y = shadowMapIndex
	// metadata.z = intensity
};

layout(set = 5, binding = 0) readonly buffer Lights {
	ShaderLight lights[];
};

layout(set = 6, binding = 0) readonly buffer LightSpaceMatrices {
	mat4 lightSpaceMatrices[];
};

layout(push_constant) uniform PushConstants {
	int lightCount;
	float emissiveStrength;
	float shadowBias;
} pConsts;

layout(location = 0) out vec4 oColour;

vec3 posFromDepth(float depth) {
	vec4 clipSpace = vec4(v2fTexCoord * 2.0 - 1.0, depth, 1.0);
	vec4 viewSpace = inverse(mvp.view) * inverse(mvp.projection) * clipSpace;
	vec3 worldSpace = viewSpace.xyz / viewSpace.w;
	return worldSpace;
}

float distributionFunction(vec3 normal, vec3 halfwayVector, float roughness) {
	// Normal distribution function
    float nDotH = max(dot(normal, halfwayVector), 0.0001);
    float nDotH2 = nDotH * nDotH;
    float nDotH4 = nDotH2 * nDotH2;
    float roughness2 = roughness * roughness;

    float ndf_numerator = exp((nDotH2 - 1) / (roughness2 * nDotH2));
    float ndf_denom = PI * roughness2 * nDotH4;

    float ndf = ndf_numerator / (0.0001 + ndf_denom); // Add an epsilon to denom to prevent / by 0
    return ndf;
}

vec3 fresnel(float metalness, vec3 halfwayVector, vec3 viewDir) {
	// Fresnel
    // Specular base reflectivity
    vec3 f0 = (1 - metalness) * vec3(0.04) + (metalness * subpassLoad(gBuffer2).rgb);
	float base = max(1 - dot(halfwayVector, viewDir), 0.001);
    vec3 fresnel = f0 + (1 - f0) * pow(base, 5.0);
    return fresnel;
}

float geometryFunction(vec3 normal, vec3 halfwayVector, vec3 viewDir, vec3 lightDir) {
	// Geometry function
    float termLeft = 2 * (max(0, dot(normal, halfwayVector)) * max(0, dot(normal, viewDir)) / dot(viewDir, halfwayVector));
    float termRight = 2 * (max(0, dot(normal, halfwayVector)) * max(0, dot(normal, lightDir)) / dot(viewDir, halfwayVector));

    float geometry = min(1, min(termLeft, termRight));
    return geometry;    
}

vec3 brdf(vec3 lightDir, vec3 viewDir, vec3 normal, float shadow) {
	vec3 halfwayVector = normalize(viewDir + lightDir);

	float metalness = subpassLoad(gBuffer3).a;
	float roughness_sqrt = subpassLoad(gBuffer2).a;
	float roughness = roughness_sqrt * roughness_sqrt;

	float ndf = distributionFunction(normal, halfwayVector, roughness);
	vec3 fresnel = fresnel(metalness, halfwayVector, viewDir);
	float geometry = geometryFunction(normal, halfwayVector, viewDir, lightDir);

	float specular_denom = 4 * max(dot(normal, viewDir), 0.0) * max(dot(normal, lightDir), 0.0);

	vec3 diffuse = (subpassLoad(gBuffer2).rgb / PI) * (vec3(1.0) - fresnel) * (1 - metalness);
	vec3 specular = (ndf * fresnel * geometry) / (0.0001 + specular_denom);

	vec3 ret;
	// If shadow is not 1.0 then the fragment is in shadow and has no
	// direction line of sight to the light and should not contribute any
	// specular to final colour
	if (shadow < 1.0) {
		ret = diffuse * shadow;
	} else {
		ret = diffuse + specular;
	}

	return ret;
}

const mat4 biasMat = mat4( 
	0.5, 0.0, 0.0, 0.0,
	0.0, 0.5, 0.0, 0.0,
	0.0, 0.0, 1.0, 0.0,
	0.5, 0.5, 0.0, 1.0);

float calculateShadow(ShaderLight light, vec3 pos) {
	float shadow = 0.0;

	int lightType = light.metadata.x;
	int shadowMapIndex = light.metadata.y;

	switch (lightType) {
	case 0: // Point light
		vec3 lightToFrag = pos - light.position;

		float currentDepth = length(lightToFrag) / planes.far;
		vec3 dir = normalize(lightToFrag);

		shadow = texture(pointLightShadows, vec4(dir, shadowMapIndex), currentDepth - SHADOW_BIAS);
		break;
	case 1: // Directional light
		mat4 lightSpaceMatrix = biasMat * lightSpaceMatrices[shadowMapIndex];

		vec4 lightSpacePos = lightSpaceMatrix * vec4(pos, 1.0);
		vec3 shadowCoord = lightSpacePos.xyz / lightSpacePos.w;

		shadow = texture(sunShadow, shadowCoord);
		break;
	case 2: // Spot light
		break;
	}

	return shadow;
}

void main() {
    float depth = subpassLoad(inputDepth).x;
	
    // Fragments with depth of 1 are fragments that weren't drawn
    // to in the previous pass, without this, the 'skybox' would be
    // black instead of the clear color
    if (depth == 1.0) discard;
    
    // Get world space vertex position from depth buffer
    vec3 pos = posFromDepth(depth);

	vec3 normal = subpassLoad(gBuffer1).rgb;
	// Map normals from [0, 1] (gBuffer format is UNORM) back to [-1, 1]
	normal = normal * 2.0 - 1.0;

	vec3 ambient = vec3(0.03) * subpassLoad(gBuffer2).rgb;
	vec3 totalLight = ambient;

    // Iterate over all lights
    for (int i = 0; i < pConsts.lightCount; i++) {

		vec3 lightPos = lights[i].position;
		float distToLight = length(lightPos - pos);
		vec3 lightDir = normalize(lightPos - pos);
        
		float attenuation;
		if (lights[i].metadata.x == 1) {
			// Directional lights have no attenuation
			attenuation = 1;
			// Light dir should be parallel for every fragment for directional lights
			lightDir = -lights[i].direction;
		} else {
			// Keep point and spot lights with squared attenuation
			attenuation = 1 / (distToLight * distToLight);
		}

		vec3 viewDir = normalize(mvp.camPos.xyz - pos);

		float shadow = calculateShadow(lights[i], pos);

		vec3 brdfVal = brdf(lightDir, viewDir, normal, shadow) * lights[i].metadata.z;
		float NdotL = max(dot(normal, lightDir), 0.0001);

		totalLight += (brdfVal * NdotL) * lights[i].colour * attenuation;
	}

	vec3 emissive = subpassLoad(gBuffer3).rgb;
	totalLight += emissive * pConsts.emissiveStrength;

    oColour = vec4(totalLight, 1.0);
}