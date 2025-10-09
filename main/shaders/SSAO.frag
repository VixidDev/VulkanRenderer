#version 450

layout(location = 0) in vec2 v2fTexCoord;

layout(set = 0, binding = 0) uniform ProjectiveUniforms {
	mat4 projection;
	mat4 invProjection;
} projections;

layout(set = 1, binding = 0) uniform SSAOUniform {
	vec4 samples[64];
	float radius;
} ssaoUniform;

layout(set = 2, binding = 0) uniform sampler2D uDepth;
layout(set = 2, binding = 1) uniform sampler2D uNormal;
layout(set = 2, binding = 2) uniform sampler2D uNoise;

layout(location = 0) out float ao;

vec3 getVSPosFromDepth(vec2 uv) {
	float depth = texture(uDepth, uv).r;
	float x = v2fTexCoord.x * 2.0 - 1.0;
	float y = (1.0 - v2fTexCoord.y) * 2.0 - 1.0;
	//vec2 xy = v2fTexCoord * 2.0 - 1.0;
	vec4 pos = vec4(x, y, depth, 1.0);
	vec4 posVS = projections.invProjection * pos;
	return posVS.xyz / posVS.w;
}

// Credit: https://ajweeks.com/blog/2019/05/11/SSAO/
void main() {
	float depth = texture(uDepth, v2fTexCoord).r;

	if (depth == 1.0) discard;

	vec3 normal = texture(uNormal, v2fTexCoord).rgb * 2.0 - 1.0;
	vec3 posVS = getVSPosFromDepth(v2fTexCoord);

	// Tile noise over screen
	vec2 depthTexSize = textureSize(uDepth, 0);
	vec2 noiseTexSize = textureSize(uNoise, 0);
	float scale = 1.0; // Render at 1:1 size for now
	vec2 noiseUV = vec2(depthTexSize.x / noiseTexSize.x, depthTexSize.y / noiseTexSize.y) * v2fTexCoord * scale;
	vec3 randomVec = vec3(texture(uNoise, v2fTexCoord).xy, 0.0);

	// Use Gram-Schmidt process to create an orthogonalise basis
	vec3 tangent = normalize(randomVec - normal * dot(randomVec, normal));
	vec3 bitangent = cross(tangent, normal);
	mat3 TBN = mat3(tangent, bitangent, normal);

	float occlusion = 0.0;
	int sampleCount = 0;
	for (uint i = 0; i < 64; i++) {
		vec3 samplePos = TBN * ssaoUniform.samples[i].xyz;
		samplePos = posVS + samplePos * ssaoUniform.radius;

		vec4 offset = vec4(samplePos, 1.0);
		offset = projections.projection * offset;
		offset.xy /= offset.w;
		offset.xy = offset.xy * 0.5 + 0.5;
		offset.y = 1.0 - offset.y;

		vec3 reconPos = getVSPosFromDepth(offset.xy);
		vec3 sampledNormal = normalize(texture(uNormal, offset.xy).xyz * 2.0 - 1.0);
		if (dot(sampledNormal, normal) > 0.99) {
			sampleCount++;
		} else {
			float rangeCheck = smoothstep(0.0, 1.0, ssaoUniform.radius / abs(reconPos.z - samplePos.z - 0.01));
			occlusion += (reconPos.z <= samplePos.z - 0.01 ? 1.0 : 0.0) * rangeCheck;
			sampleCount++;
		}
	}

	occlusion = 1.0 - (occlusion / float(max(sampleCount, 1)));
	ao = occlusion;
}