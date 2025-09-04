#version 450

#define PI 3.14159265359

layout(location = 0) in vec3 v2fPosition;
layout(location = 1) in vec2 v2fTexCoord;
layout(location = 2) in vec4 v2fLightSpacePosition;
layout(location = 3) in mat3 v2fTBN;

layout(set = 0, binding = 0) uniform MVP {
	mat4 projection;
	mat4 view;
	vec4 camPos;
} mvp;

layout(set = 1, binding = 0) uniform sampler2D uTexColor;
layout(set = 1, binding = 1) uniform sampler2D uMetalness;
layout(set = 1, binding = 2) uniform sampler2D uRoughness;
layout(set = 1, binding = 3) uniform sampler2D uAlphaMask;
layout(set = 1, binding = 4) uniform sampler2D uNormalMap;

layout(set = 3, binding = 0) uniform sampler2DShadow shadowMap;

layout(location = 0) out vec4 oColour;

float distributionFunction(vec3 normal, vec3 halfwayVector, float roughness) {
	// Normal distribution function
    float nDotH = max(dot(normal, halfwayVector), 0.0001f);
    float nDotH2 = nDotH * nDotH;
    float nDotH4 = nDotH2 * nDotH2;
    float roughness2 = roughness * roughness;

    float ndf_numerator = exp((nDotH2 - 1) / (roughness2 * nDotH2));
    float ndf_denom = PI * roughness2 * nDotH4;

    float ndf = ndf_numerator / (0.0001f + ndf_denom); // Add an epsilon to denom to prevent / by 0
    return ndf;
}

vec3 fresnel(float metalness, vec3 halfwayVector, vec3 viewDir) {
	// Fresnel
    // Specular base reflectivity
    vec3 f0 = (1 - metalness) * vec3(0.04f) + (metalness * texture(uTexColor, v2fTexCoord).rgb);
    vec3 fresnel = f0 + (1 - f0) * pow((1 - dot(halfwayVector, viewDir)), 5.0f);
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

	float metalness = texture(uMetalness, v2fTexCoord).r;
	float roughness_sqrt = texture(uRoughness, v2fTexCoord).r;
	float roughness = roughness_sqrt * roughness_sqrt;

	float ndf = distributionFunction(normal, halfwayVector, roughness);
	vec3 fresnel = fresnel(metalness, halfwayVector, viewDir);
	float geometry = geometryFunction(normal, halfwayVector, viewDir, lightDir);

	float specular_denom = 4 * max(dot(normal, viewDir), 0.0f) * max(dot(normal, lightDir), 0.0f);

	vec3 diffuse = (texture(uTexColor, v2fTexCoord).rgb / PI) * (vec3(1.0f) - fresnel) * (1 - metalness);
	vec3 specular = (ndf * fresnel * geometry) / (0.0001f + specular_denom);

	vec3 ret;
	if (shadow < 1.0f) {
		ret = diffuse * shadow;
	} else {
		ret = diffuse + specular;
	}

	return ret;
}

void main() {
	vec3 normal = v2fTBN * normalize(texture(uNormalMap, v2fTexCoord).rgb * 2.0f - 1.0f);

	vec3 lightPos = vec3(-0.2972f, 7.3100f, -11.9532f);

	vec3 lightDir = normalize(lightPos - v2fPosition);
	vec3 viewDir = normalize(mvp.camPos.rgb - v2fPosition);

	vec3 ambient = vec3(0.03f) * texture(uTexColor, v2fTexCoord).rgb;

	float alphaValue = texture(uAlphaMask, v2fTexCoord).a;
	if (alphaValue < 0.5f) discard;

	float shadow = max(texture(shadowMap, v2fLightSpacePosition.xyz / v2fLightSpacePosition.w), 0.1f);

	vec3 brdfVal = brdf(lightDir, viewDir, normal, shadow) * 100;
	vec3 lightCol = vec3(1.0f);
	float NdotL = max(dot(normal, lightDir), 0.0001f);
	float attenuation = 1 / pow(length(lightPos - v2fPosition), 2);

	vec3 colour = ambient + (brdfVal * lightCol * NdotL) * attenuation;

	oColour = vec4(colour, 1.0f);
}