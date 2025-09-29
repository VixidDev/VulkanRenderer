#version 450

layout(location = 0) in vec2 v2fTexCoord;

layout(set = 0, binding = 0) uniform sampler2D brightness;

layout(location = 0) out vec4 oBlur;

layout(push_constant) uniform PushConstant {
	int direction; // 0 = horizontal, 1 = vertical
} pConsts;

const float weight[5] = float[5](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

void main() {
	vec2 texelSize = 1.0 / textureSize(brightness, 0);

	vec3 result = texture(brightness, v2fTexCoord).rgb * weight[0];

	if (pConsts.direction == 0) {
		for (int i = 0; i < 5; i++) {
			result += texture(brightness, v2fTexCoord + vec2(texelSize.x * i, 0.0)).rgb * weight[i];
			result += texture(brightness, v2fTexCoord - vec2(texelSize.x * i, 0.0)).rgb * weight[i];
		}
	} else {
		for (int i = 0; i < 5; i++) {
			result += texture(brightness, v2fTexCoord + vec2(0.0, texelSize.y * i)).rgb * weight[i];
			result += texture(brightness, v2fTexCoord - vec2(0.0, texelSize.y * i)).rgb * weight[i];
		}
	}

	oBlur = vec4(result, 1.0);
}