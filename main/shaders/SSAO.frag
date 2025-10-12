#version 450

#define KERNEL_SIZE 32

layout(location = 0) in vec2 v2fTexCoord;

layout(set = 0, binding = 0) uniform ProjectiveUniforms {
	mat4 projection;
	mat4 invProjection;
} projections;

layout(set = 1, binding = 0) uniform SSAOUniform {
	vec4 samples[KERNEL_SIZE];
	float radius;
} ssaoUniform;

layout(set = 2, binding = 0) uniform sampler2D uDepth;
layout(set = 2, binding = 1) uniform sampler2D uNormal; // View-space normals
layout(set = 2, binding = 2) uniform sampler2D uNoise;

layout(location = 0) out float ao;

vec3 getVSPosFromDepth(vec2 uv) {
	float depth = texture(uDepth, uv).r;
	vec2 xy = uv * 2.0 - 1.0;
	vec4 pos = vec4(xy.x, xy.y, depth, 1.0);
	vec4 posVS = projections.invProjection * pos;
	return posVS.xyz / posVS.w;
}

// Credit: https://ajweeks.com/blog/2019/05/11/SSAO/
void main() {
	float depth = texture(uDepth, v2fTexCoord).r;
	
	if (depth == 0.0) {
		ao = 1.0;
		return;
	}
	
	vec3 normal = normalize(texture(uNormal, v2fTexCoord).rgb * 2.0 - 1.0);
	vec3 fragPos = getVSPosFromDepth(v2fTexCoord);

	// Tile noise over screen
	vec2 depthTexSize = textureSize(uDepth, 0);
	vec2 noiseTexSize = textureSize(uNoise, 0);
	float scale = 0.5;
	vec2 noiseUV = vec2(float(depthTexSize.x) / float(noiseTexSize.x), float(depthTexSize.y) / float(noiseTexSize.y)) * v2fTexCoord * scale;
	vec3 randomVec = vec3(texture(uNoise, noiseUV).xy, 0.0);

	// Use Gram-Schmidt process to create an orthogonalise basis
	vec3 tangent = normalize(randomVec - normal * dot(randomVec, normal));
	vec3 bitangent = cross(tangent, normal);
	mat3 TBN = mat3(tangent, bitangent, normal);

	float occlusion = 0.0;
	const float bias = 0.025;
	int sampleCount = 0;
	for (uint i = 0; i < KERNEL_SIZE; i++) {
		vec3 samplePos = TBN * ssaoUniform.samples[i].xyz;
		samplePos = fragPos + samplePos * ssaoUniform.radius;

		vec4 offset = vec4(samplePos, 1.0);
		offset = projections.projection * offset;
		offset.xy /= offset.w;
		offset.xy = offset.xy * 0.5 + 0.5;

		vec3 reconPos = getVSPosFromDepth(offset.xy);
		float rangeCheck = smoothstep(0.0, 1.0, ssaoUniform.radius / abs(fragPos.z - reconPos.z - bias));
		occlusion += (reconPos.z >= samplePos.z + bias ? 1.0 : 0.0) * rangeCheck;
	}

	occlusion = 1.0 - (occlusion / KERNEL_SIZE);
	ao = occlusion;
}