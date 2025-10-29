#define PI 3.14159265359

struct ShaderLight {
	vec4 positionAndLightType;    // xyz: light position, w: light type
	vec4 directionAndMapIndex;    // xyz: light direction, w: shadow map index
	vec4 colourAndIntensity;      // xyz: light colour, w: light intensity
	vec4 extra; // x: spot light inner cone angle, y: spot light outer cone angle, z: light space matrix index, w: is shadow casting
};

float BeckmannDistribution(float nDotH, float a2) {
    float nDotH2 = nDotH * nDotH;
    float nDotH4 = nDotH2 * nDotH2;

    float numerator = exp((nDotH2 - 1) / (a2 * nDotH2));
    float denomenator = PI * a2 * nDotH4;

    return numerator / (denomenator + 0.0001); // Add an epsilon to denom to prevent / by 0
}

float GGXTrowbridgeReitz(float nDotH, float a2) {
	float nDotH2 = nDotH * nDotH;

	float denom = (nDotH2 * (a2 - 1.0) + 1.0);
	denom = PI * denom * denom;

	return a2 / (denom + 0.0001);
}

vec3 FresnelSchlick(float cosTheta, vec3 F0) {
	float Fc = 1.0 - cosTheta;
	float Fc2 = Fc * Fc;
	float Fc5 = Fc2 * Fc2 * Fc;
	return F0 + (1 - F0) * Fc5;
}

float CookTorranceGeometry(vec3 viewDir, vec3 halfwayVector, float nDotH, float nDotV, float nDotL) {
	float vDotH = dot(viewDir, halfwayVector);

    float termLeft  = (2 * max(0, nDotH) * max(0, nDotV)) / max(0, vDotH);
    float termRight = (2 * max(0, nDotH) * max(0, nDotL)) / max(0, vDotH);

    return min(1, min(termLeft, termRight));
}

// https://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf
float GeometrySchlickGGX(float nDot, float a) {
	float r = a + 1.0;
	float k = (r * r) / 8.0;

	float denom = nDot * (1.0 - k) + k;

	return nDot / denom;
}

float GeometrySmith(float nDotL, float nDotV, float a) {
	float ggx1 = GeometrySchlickGGX(max(nDotL, 0.0001), a);
	float ggx2 = GeometrySchlickGGX(max(nDotV, 0.0001), a);
	return ggx1 * ggx2;
}

vec3 CookTorranceBRDF(vec3 lightDir, vec3 viewDir, vec3 normal, float nDotV, float metalness, float roughness, 
					  float a, float a2, vec3 F0, vec3 albedo, vec3 radiance, float shadow) 
{
	vec3 halfwayVector = normalize(viewDir + lightDir);
	float hDotV = dot(halfwayVector, viewDir);
	float nDotH = dot(normal, halfwayVector);
	float nDotL = dot(normal, lightDir);

	float D = BeckmannDistribution(max(nDotH, 0.0), a2);
	vec3  F = FresnelSchlick(hDotV, F0);
	float G = GeometrySmith(nDotL, nDotV, roughness);

	float specDenom = 4 * max(nDotV, 0.0) * max(nDotL, 0.0);
	vec3 specular = (D * F * G) / max(specDenom, 0.0001);

	vec3 kS = F;
	vec3 kD = vec3(1.0) - kS;
	kD *= 1.0 - metalness;

	vec3 diffuse = kD * albedo / PI;
	
	vec3 combined = (diffuse + specular) * shadow;

	return combined * radiance * max(nDotL, 0.0);
}