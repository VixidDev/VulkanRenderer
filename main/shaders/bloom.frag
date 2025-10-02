#version 450

layout(location = 0) in vec2 v2fTexCoord;

layout(set = 0, binding = 0) uniform sampler2D brightness;

layout(location = 0) out vec4 oBlur;

layout(push_constant) uniform PushConstant {
	int direction; // 0 = horizontal, 1 = vertical
} pConsts;

const float offsets[11] = float[11](
	0.0, 1.4680851063829785, 3.4255319148936167, 5.382978723404256, 7.340425531914894, 
	9.297872340425531, 11.25531914893617, 13.21276595744681, 15.170212765957446, 17.127659574468087, 19.085106382978722);
const float weight[11] = float[11](
	0.11700408787775982, 0.2108023649930973, 0.13873318038007257, 0.06492439475914726, 0.02136222021107426, 
	0.0048550500479714225, 0.0007425370661603353, 7.358475430417737e-05, 4.46870977555733e-06, 1.5259008989707957e-07, 
	2.5347191012803915e-09);

void main() {
	vec2 texelSize = 1.0 / textureSize(brightness, 0);

	vec3 result = texture(brightness, v2fTexCoord).rgb * weight[0];

	if (pConsts.direction == 0) {
		for (int i = 1; i < 11; ++i) {
			result += texture(brightness, v2fTexCoord + vec2(texelSize.x * offsets[i], 0.0)).rgb * weight[i];
			result += texture(brightness, v2fTexCoord - vec2(texelSize.x * offsets[i], 0.0)).rgb * weight[i];
		}
	} else {
		for (int i = 1; i < 11; ++i) {
			result += texture(brightness, v2fTexCoord + vec2(0.0, texelSize.y * offsets[i])).rgb * weight[i];
			result += texture(brightness, v2fTexCoord - vec2(0.0, texelSize.y * offsets[i])).rgb * weight[i];
		}
	}

	oBlur = vec4(result, 1.0);
}