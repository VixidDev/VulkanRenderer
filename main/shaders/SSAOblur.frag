#version 450

layout(location = 0) in vec2 v2fTexCoord;

layout(set = 0, binding = 0) uniform sampler2D uDepth;
layout(set = 0, binding = 1) uniform sampler2D uNormal;
layout(set = 0, binding = 2) uniform sampler2D uSSAO;

layout(set = 1, binding = 0) uniform ClipPlanes {
	float far;
	float near;
} planes;

layout(location = 0) out float oColour;

layout(push_constant) uniform PushConstant {
	int direction; // 0 = horizontal, 1 = vertical
	int radius;
	float depthThreshold;
	float normalThreshold;
} pConsts;

float lineariseDepth(float depth) {
	return (2.0 * planes.near * planes.far) / (planes.far + planes.near - (2.0 * depth - 1.0) * (planes.far - planes.near));
}

// Adapted from: https://github.com/ajweeks/FlexEngine/blob/development/FlexEngine/resources/shaders/vk_ssao_blur.frag
void main() {
	float depth = lineariseDepth(texture(uDepth, v2fTexCoord).r);
	vec3 normal = normalize(texture(uNormal, v2fTexCoord).xyz * 2.0 - 1.0);

	vec2 texSize = textureSize(uDepth, 0);

	vec2 texelOffset;
	if (pConsts.direction == 0) {
		texelOffset = vec2(2.0, 0.0) / texSize;
	} else {
		texelOffset = vec2(0.0, 2.0) / texSize;
	}

	int hits = 0;
	float sum = 0.0;
	for (int i = -pConsts.radius; i <= pConsts.radius; i++) {
		vec2 offset = texelOffset * float(i);
		float sampleDepth = lineariseDepth(texture(uDepth, v2fTexCoord + offset).r);
		vec3 sampleNormal = normalize(texture(uNormal, v2fTexCoord + offset).xyz * 2.0 - 1.0);

		if (abs(depth - sampleDepth) < pConsts.depthThreshold && dot(normal, sampleNormal) > pConsts.normalThreshold) {
			sum += texture(uSSAO, v2fTexCoord + offset).r;
			hits++;
		}
	}

	oColour = clamp(sum / float(hits), 0.0, 1.0);
}