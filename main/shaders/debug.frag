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

layout(set = 2, binding = 0) uniform ClipPlanes {
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

layout(set = 3, binding = 0) readonly buffer Lights {
	ShaderLight lights[];
};

layout(location = 0) out vec4 oColour;

layout(push_constant) uniform PushConstants {
    int lightCount;
    int debugState;
} pConsts;

float lineariseDepth(float depth) {
	return (2.0 * planes.near * planes.far) / (planes.far + planes.near - (2.0 * depth - 1.0) * (planes.far - planes.near));
}

// Colors for mipmaps
// Colorblind friendly colors https://davidmathlogic.com/colorblind/
const vec3 colours[7] = vec3[7](
    vec3(17.0, 119.0, 51.0) / 255,   // Murky green
    vec3(136.0, 34.0, 85.0) / 255,   // Purple
    vec3(0.0, 114.0, 178.0) / 255,   // Blue
    vec3(204.0, 121.0, 167.0) / 255, // Pink
    vec3(0.0, 158.0, 115.0) / 255,   // Turquoise
    vec3(213.0, 94.0, 0.0) / 255,    // Orange
    vec3(240.0, 228.0, 66.0) / 255   // Yellow
);

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

float geometryFunction(vec3 normal, vec3 halfwayVector, vec3 viewDir, vec3 lightDir) {
	// Geometry function
    float termLeft = 2 * (max(0, dot(normal, halfwayVector)) * max(0, dot(normal, viewDir)) / dot(viewDir, halfwayVector));
    float termRight = 2 * (max(0, dot(normal, halfwayVector)) * max(0, dot(normal, lightDir)) / dot(viewDir, halfwayVector));

    float geometry = min(1, min(termLeft, termRight));
    return geometry;    
}

vec3 fresnel(float metalness, vec3 halfwayVector, vec3 viewDir) {
	// Fresnel
    // Specular base reflectivity
    vec3 f0 = (1 - metalness) * vec3(0.04f) + (metalness * texture(uTexColour, v2fTexCoord).rgb);
	float base = max(1 - dot(halfwayVector, viewDir), 0.001);
    vec3 fresnel = f0 + (1 - f0) * pow(base, 5.0);
    return fresnel;
}

void main() {
    vec3 normal;
	if (v2fFallbackNormal.w == 1.0) {
		normal = v2fFallbackNormal.xyz;
	} else {
		normal = v2fTBN * normalize(texture(uNormalMap, v2fTexCoord).rgb * 2.0 - 1.0);
	}

    vec3 total = vec3(0.0);

    switch(pConsts.debugState) {
        // Normals
        case 0:
            oColour = vec4(normal, 1.0);
            break;
        // Mipmap levels
        case 1:
            float mipmapLevel = textureQueryLod(uTexColour, v2fTexCoord).x;

            vec3 floorColor = colours[int(floor(mipmapLevel)) % 7];
            vec3 ceilColor = colours[int(ceil(mipmapLevel)) % 7];

            oColour = vec4(mix(floorColor, ceilColor, fract(mipmapLevel)), 1.0);
            break;
        // Linear fragment depth
        case 2:
            oColour = vec4(vec3(lineariseDepth(gl_FragCoord.z) / 100.0), 1.0);
            break;
        // Partial derivatives
        case 3:
            float depth = lineariseDepth(gl_FragCoord.z);

            float dx = dFdx(depth) * 5;
            float dy = dFdy(depth) * 5;

            // float dx = abs(dFdx(gl_FragCoord.z)) * 200;
            // float dy = abs(dFdy(gl_FragCoord.z)) * 200;

            oColour = vec4(dx, dy, 0.0, 1.0);
            break;
        // PBR distribution function
        case 4:
            for (int i = 0; i < pConsts.lightCount; i++) {
                vec3 lightPos = lights[i].position;
                vec3 lightDir = normalize(lightPos - v2fPosition);
                vec3 viewDir = normalize(mvp.camPos.rgb - v2fPosition);
                vec3 halfway = normalize(viewDir + lightDir);

                float roughnessSqrt = texture(uRoughness, v2fTexCoord).r;
                float roughness = roughnessSqrt * roughnessSqrt;

                float distribution = distributionFunction(normal, halfway, roughness);
                total += vec3(distribution);
            }
            
            oColour = vec4(total, 1.0);
            break;
        // PBR geometry function
        case 5:
            for (int i = 0; i < pConsts.lightCount; i++) {
                vec3 lightPos = lights[i].position;
                vec3 lightDir = normalize(lightPos - v2fPosition);
                vec3 viewDir = normalize(mvp.camPos.rgb - v2fPosition);
                vec3 halfway = normalize(viewDir + lightDir);

                float roughnessSqrt = texture(uRoughness, v2fTexCoord).r;
                float roughness = roughnessSqrt * roughnessSqrt;

                float geometry = geometryFunction(normal, halfway, viewDir, lightDir);
                total += vec3(geometry);
            }

            oColour = vec4(total, 1.0);
            break;
        // PBR fresnel function
        case 6:
            for (int i = 0; i < pConsts.lightCount; i++) {
                vec3 lightPos = lights[i].position;
                vec3 lightDir = normalize(lightPos - v2fPosition);
                vec3 viewDir = normalize(mvp.camPos.rgb - v2fPosition);
                vec3 halfway = normalize(viewDir + lightDir);

                float roughnessSqrt = texture(uRoughness, v2fTexCoord).r;
                float roughness = roughnessSqrt * roughnessSqrt;

                float metalness = texture(uMetalness, v2fTexCoord).r;
                vec3 F = fresnel(metalness, halfway, viewDir);
                total += F;
            }

            oColour = vec4(total, 1.0);
            break;
        default:
            oColour = texture(uTexColour, v2fTexCoord).rgba;
    }

}
