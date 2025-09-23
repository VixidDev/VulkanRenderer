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

struct ShaderLight {
	vec3 position;
	vec3 direction;
	vec3 colour;
	ivec3 metadata;
	// metadata.x = lightType // 0 - Point Light, 1 - Directional Light, 2 - Spot light
	// metadata.y = shadowMapIndex
	// metadata.z = intensity
};

layout(set = 2, binding = 0) readonly buffer Lights {
	ShaderLight lights[];
};

layout(push_constant) uniform PushConstants {
	int lightCount;
} pConsts;

layout(location = 0) out vec4 oColour;

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
    vec3 f0 = (1 - metalness) * vec3(0.04f) + (metalness * texture(uTexColour, v2fTexCoord).rgb);
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

vec3 brdf(vec3 lightDir, vec3 viewDir, vec3 normal) {
	vec3 halfwayVector = normalize(viewDir + lightDir);

	float metalness = texture(uMetalness, v2fTexCoord).r;
	float roughness_sqrt = texture(uRoughness, v2fTexCoord).r;
	float roughness = roughness_sqrt * roughness_sqrt;

	float ndf = distributionFunction(normal, halfwayVector, roughness);
	vec3 fresnel = fresnel(metalness, halfwayVector, viewDir);
	float geometry = geometryFunction(normal, halfwayVector, viewDir, lightDir);

	float specular_denom = 4 * max(dot(normal, viewDir), 0.0) * max(dot(normal, lightDir), 0.0);

	vec3 diffuse = (texture(uTexColour, v2fTexCoord).rgb / PI) * (vec3(1.0) - fresnel) * (1 - metalness);
	vec3 specular = (ndf * fresnel * geometry) / (0.0001 + specular_denom);

	return diffuse + specular;
}

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

	vec3 ambient = vec3(0.03) * texture(uTexColour, v2fTexCoord).rgb;
	vec3 totalLight = ambient;

	// Iterate over all lights
	for (int i = 0; i < pConsts.lightCount; i++) {

		vec3 lightPos = lights[i].position;
		float distToLight = length(lightPos - v2fPosition);
		vec3 lightDir = normalize(lightPos - v2fPosition);

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

		vec3 viewDir = normalize(mvp.camPos.rgb - v2fPosition);

		vec3 brdfVal = brdf(lightDir, viewDir, normal) * lights[i].metadata.z;
		float NdotL = max(dot(normal, lightDir), 0.0001);

		totalLight += (brdfVal * NdotL) * lights[i].colour * attenuation;
	}

	oColour = vec4(totalLight, 1.0);
}