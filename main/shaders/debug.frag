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

layout(set = 2, binding = 0) uniform ClipPlanes {
	float far;
	float near;
} planes;

layout(set = 3, binding = 0) uniform sampler2D uTexColour;
layout(set = 3, binding = 1) uniform sampler2D uMetalness;
layout(set = 3, binding = 2) uniform sampler2D uRoughness;
layout(set = 3, binding = 3) uniform sampler2D uAlphaMask;
layout(set = 3, binding = 4) uniform sampler2D uNormalMap;
layout(set = 3, binding = 5) uniform sampler2D uEmissive;

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

void main() {
    vec3 normal;
	if (v2fFallbackNormal.w == 1.0) {
		normal = normalize(v2fFallbackNormal.xyz);
	} else {
		vec3 tangentNormal = texture(uNormalMap, v2fTexCoord).rgb;
		tangentNormal = tangentNormal * 2.0 - 1.0;
		normal = normalize(v2fTBN * tangentNormal);
	}

    vec3 lightPos;
    vec3 lightDir;
    vec3 viewDir;
    float roughness;
    vec3 halfway;

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
            oColour = vec4(vec3((lineariseDepth(gl_FragCoord.z) - planes.near) / (planes.far - planes.near)), 1.0);
            break;
        // PBR distribution function
        case 3:
            lightPos = lights[0].positionAndLightType.xyz;
            lightDir = normalize(lightPos - v2fPosition);
            viewDir = normalize(mvp.camPos.rgb - v2fPosition);
            halfway = normalize(viewDir + lightDir);

            float nDotH = dot(normal, halfway);

            roughness = texture(uRoughness, v2fTexCoord).r;
            float a = roughness * roughness;
            float a2 = a * a;

            float distribution = BeckmannDistribution(max(nDotH, 0.0), a2);
            
            oColour = vec4(vec3(distribution), 1.0);
            break;
        // PBR geometry function
        case 4:
            lightPos = lights[0].positionAndLightType.xyz;
            lightDir = normalize(lightPos - v2fPosition);
            viewDir = normalize(mvp.camPos.rgb - v2fPosition);

            float nDotL = dot(normal, lightDir);
            float nDotV = dot(normal, viewDir);

            roughness = texture(uRoughness, v2fTexCoord).r;

            float geometry = GeometrySmith(nDotL, nDotV, roughness);

            oColour = vec4(vec3(geometry), 1.0);
            break;
        // PBR fresnel function
        case 5:
            lightPos = lights[0].positionAndLightType.xyz;
            lightDir = normalize(lightPos - v2fPosition);
            viewDir = normalize(mvp.camPos.rgb - v2fPosition);
            halfway = normalize(viewDir + lightDir);

            float hDotV = dot(halfway, viewDir);

            float metalness = texture(uMetalness, v2fTexCoord).r;
            vec3 albedo = texture(uTexColour, v2fTexCoord).rgb;

            vec3 F0 = vec3(0.04);
            F0 = mix(F0, albedo, metalness);

            vec3 F = FresnelSchlick(hDotV, F0);

            oColour = vec4(F, 1.0);
            break;
        default:
            oColour = texture(uTexColour, v2fTexCoord).rgba;
    }

}
