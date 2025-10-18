#define PI 3.14159265359

struct ShaderLight {
	vec4 positionAndLightType; // xyz: light position, w: light type
	vec4 directionAndMapIndex; // xyz: light direction, w: shadow map index
	vec4 colourAndIntensity;   // xyz: light colour, w: light intensity
	//vec2 spotLightInfo;        // x: spot light inner cone angle, y: spot light outer cone angle
};

float BeckmannDistribution(float nDotH, float roughness) {
    float nDotH2 = nDotH * nDotH;
    float nDotH4 = nDotH2 * nDotH2;
    float roughness2 = roughness * roughness;

    float numerator = exp((nDotH2 - 1) / (roughness2 * nDotH2));
    float denomenator = PI * roughness2 * nDotH4;

    return numerator / (denomenator + 0.0001); // Add an epsilon to denom to prevent / by 0
}

vec3 FresnelSchlick(float cosTheta, vec3 F0) {
	return F0 + (1 - F0) * pow(1.0 - cosTheta, 5.0);
}

float CookTorranceGeometry(vec3 viewDir, vec3 halfwayVector, float nDotH, float nDotV, float nDotL) {
	float vDotH = dot(viewDir, halfwayVector);

    float termLeft  = (2 * max(0, nDotH) * max(0, nDotV)) / max(0, vDotH);
    float termRight = (2 * max(0, nDotH) * max(0, nDotL)) / max(0, vDotH);

    return min(1, min(termLeft, termRight));
}

vec3 CookTorranceBRDF(vec3 lightDir, vec3 viewDir, vec3 normal, float metalness, float roughnessSqrt, 
					  vec3 F0, vec3 albedo, vec3 radiance, float shadow) 
{
	vec3 halfwayVector = normalize(viewDir + lightDir);
	float hDotV = dot(halfwayVector, viewDir);
	float nDotH = dot(normal, halfwayVector);
	float nDotV = dot(normal, viewDir);
	float nDotL = dot(normal, lightDir);

	float roughness = roughnessSqrt * roughnessSqrt;

	float D = BeckmannDistribution(max(nDotH, 0.0001), roughness);
	vec3  F = FresnelSchlick(max(hDotV, 0.0), F0);
	float G = CookTorranceGeometry(viewDir, halfwayVector, nDotH, nDotV, nDotL);

	float specDenom = 4 * max(nDotV, 0.0) * max(nDotL, 0.0);
	vec3 specular = (D * F * G) / max(specDenom, 0.0001);

	vec3 kS = F;
	vec3 kD = vec3(1.0) - kS;
	kD *= 1.0 - metalness;

	vec3 diffuse = (albedo / PI) * kD;
	
	vec3 combined = (diffuse + specular) * shadow;

	return combined * radiance * max(nDotL, 0.0);
}