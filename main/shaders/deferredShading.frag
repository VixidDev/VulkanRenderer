#version 450

#define PI 3.14159265359

#define LIGHT_POINTS 21

layout(location = 0) in vec2 v2fTexCoord;

layout(set = 0, input_attachment_index = 0, binding = 0) uniform subpassInput inputNormals; // xyz: normal w: metalness
layout(set = 0, input_attachment_index = 1, binding = 1) uniform subpassInput inputAlbedo;  // xyz: albedo w: roughness
layout(set = 0, input_attachment_index = 2, binding = 2) uniform subpassInput inputDepth;
                                                                                            // metalness and roughness only need 8 bits for storage
                                                                                            // so we are technically wasting 8 bits for roughness as
                                                                                            // the albedo texture image is RGB_16F

layout(set = 1, binding = 0) uniform UScene {
    mat4 camera;
    mat4 projection;
    mat4 projCam;
    vec4 camPos;
} uScene;

struct Light {
    vec4 lightPos;
    vec4 lightColour;
};

layout(set = 2, binding = 0) uniform Lights {
    Light light[LIGHT_POINTS];
} lights;

layout(location = 0) out vec4 oColor;

vec3 posFromDepth(float depth) {
	vec4 clipSpace = vec4(v2fTexCoord * 2.0f - 1.0f, depth, 1.0f);
	vec4 viewSpace = inverse(uScene.camera) * inverse(uScene.projection) * clipSpace;

	vec3 worldSpace = viewSpace.xyz / viewSpace.w;

	return worldSpace;
}

float DistributionFunction(vec3 normal, vec3 halfwayVector, float roughness) {
    // Normal distribution function
    float nDotH = max(dot(normal, halfwayVector), 0.0);
    float nDotH2 = nDotH * nDotH;
    float nDotH4 = nDotH2 * nDotH2;
    float roughness2 = roughness * roughness;

    float ndf_numerator = exp((nDotH2 - 1) / (roughness2 * nDotH2));
    float ndf_denom = PI * roughness2 * nDotH4;

    float ndf = ndf_numerator / (0.001f + ndf_denom); // Add an epsilon to denom to prevent / by 0
    return ndf;
}

vec3 Fresnel(float metalness, vec3 halfwayVector, vec3 viewDir) {
    // Fresnel
    // Specular base reflectivity
    vec3 f0 = (1 - metalness) * vec3(0.04) + (metalness * subpassLoad(inputAlbedo).rgb);
    vec3 fresnel = f0 + (1 - f0) * pow((1 - dot(halfwayVector, viewDir)), 5.0);
    return fresnel;
}

float GeometryFunction(vec3 normal, vec3 halfwayVector, vec3 viewDir, vec3 lightDir) {
    // Geometry function
    float termLeft = 2 * (max(0, dot(normal, halfwayVector)) * max(0, dot(normal, viewDir)) / dot(viewDir, halfwayVector));
    float termRight = 2 * (max(0, dot(normal, halfwayVector)) * max(0, dot(normal, lightDir)) / dot(viewDir, halfwayVector));

    float geometry = min(1, min(termLeft, termRight));
    return geometry;
}

vec3 brdf(vec3 lightDir, vec3 viewDir, vec3 normal) {
    vec3 halfwayVector = normalize(viewDir + lightDir);

    float metalness = subpassLoad(inputNormals).a;
    float roughness_sqrt = subpassLoad(inputAlbedo).a;
    float roughness = roughness_sqrt * roughness_sqrt;

    float ndf = DistributionFunction(normal, halfwayVector, roughness);
    vec3 fresnel = Fresnel(metalness, halfwayVector, viewDir);
    float geometry = GeometryFunction(normal, halfwayVector, viewDir, lightDir);    

    vec3 diffuse = (subpassLoad(inputAlbedo).rgb / PI) * (vec3(1.0f) - fresnel) * (1 - metalness);

    float brdf_denom = 4 * max(dot(normal, viewDir), 0.0) * max(dot(normal, lightDir), 0.0);

    return diffuse + ((ndf * fresnel * geometry) / (0.001f + brdf_denom));
}

void main() {
    float depth = subpassLoad(inputDepth).x;
	
    // Fragments with depth of 1 are fragments that weren't drawn
    // to in the previous pass, without this, the 'skybox' would be
    // black instead of the clear color
    if (depth == 1.0f) discard;
    
    // Get world space vertex position from depth buffer
    vec3 pos = posFromDepth(depth);

    // Continue with regular PBR shading
	vec3 viewDir = normalize(uScene.camPos.xyz - pos);
	vec3 normal = normalize(subpassLoad(inputNormals).xyz);

    // Ambient part
    vec3 ambient = vec3(0.03f) * subpassLoad(inputAlbedo).rgb;

	vec4 color = vec4(0.0f);

    // Apply each light's contribution
    for (int i = 0; i < LIGHT_POINTS; i++) {
        vec3 lightDir = normalize(lights.light[i].lightPos.xyz - pos);
        color += vec4(ambient.xyz + (brdf(lightDir, viewDir, normal) * 10) * lights.light[i].lightColour.rgb * (max(dot(normal, lightDir), 0.0f)), 1.0f) / pow(length(lights.light[i].lightPos.xyz - pos), 2);
    }

    oColor = color;
}