#version 450

layout(location = 0) in vec2 v2fTexCoord;

layout(set = 0, binding = 0) uniform sampler2D brightness;

layout(location = 0) out vec4 oBlur;

layout(push_constant) uniform PushConstant {
	int direction; // 0 = horizontal, 1 = vertical
} pConsts;

layout(constant_id = 0) const int KERNEL_SIZE = 0;

// Equivalent to a 5x5 tap
const float offsets2[2] = float[2](0, 1.3333333333333335);
const float weights2[2] = float[2](0.29411764705882354, 0.3529411764705882);

// Equivalent to a 9x9 tap
const float offsets3[3] = float[3](0, 1.3846153846153848, 3.230769230769231);
const float weights3[3] = float[3](0.22702702702702704, 0.3162162162162162, 0.07027027027027027);

// Equivalent to a 17x17 tap
const float offsets5[5] = float[5](0, 1.4285714285714288, 3.333333333333333, 5.238095238095238, 7.142857142857143);
const float weights5[5] = float[5](0.176204109737977, 0.2803247200376907, 0.11089769144348204, 0.019407096002609356, 0.0012684376472293698);

// Equivalent to a 25x25 tap
const float offsets7[7] = float[7](0, 1.4482758620689655, 3.379310344827586, 5.310344827586206, 7.241379310344828, 9.172413793103447, 11.103448275862068);
const float weights7[7] = float[7](0.1494460130776046, 0.25281283878961447, 0.12888497663784268, 0.03730880902674393, 0.005814359848323729, 0.00044239694498115334, 1.3612213691727795e-05);

// Equivalent to a 41x41 tap
const float offsets11[11] = float[11](0, 1.4666666666666666, 3.422222222222222, 5.377777777777777, 7.333333333333335, 9.28888888888889, 11.244444444444445, 13.200000000000001, 15.155555555555555, 17.11111111111111, 19.066666666666663);
const float weights11[11] = float[11](0.11960417871993993, 0.21450749444337053, 0.13860484256340866, 0.06270219068344678, 0.019603443524020138, 0.004149922520205876, 0.0005769945750018865, 5.037254226206946e-05, 2.5795327474174974e-06, 6.944895858431723e-08, 8.066081136389923e-10);

void main() {
	vec2 texelSize = 1.0 / textureSize(brightness, 0);

	vec3 result = vec3(0.0);

	switch (KERNEL_SIZE) {
		case 0: // 5x5 tap - offsets2 & weights2
			result = texture(brightness, v2fTexCoord).rgb * weights2[0];

			if (pConsts.direction == 0) {
				for (int i = 1; i < 2; ++i) {
					result += texture(brightness, v2fTexCoord + vec2(texelSize.x * offsets2[i], 0.0)).rgb * weights2[i];
					result += texture(brightness, v2fTexCoord - vec2(texelSize.x * offsets2[i], 0.0)).rgb * weights2[i];
				}
			} else {
				for (int i = 1; i < 2; ++i) {
					result += texture(brightness, v2fTexCoord + vec2(0.0, texelSize.y * offsets2[i])).rgb * weights2[i];
					result += texture(brightness, v2fTexCoord - vec2(0.0, texelSize.y * offsets2[i])).rgb * weights2[i];
				}
			}
			break;
		case 1: // 9x9 tap - offsets3 & weights3
			result = texture(brightness, v2fTexCoord).rgb * weights3[0];

			if (pConsts.direction == 0) {
				for (int i = 1; i < 3; ++i) {
					result += texture(brightness, v2fTexCoord + vec2(texelSize.x * offsets3[i], 0.0)).rgb * weights3[i];
					result += texture(brightness, v2fTexCoord - vec2(texelSize.x * offsets3[i], 0.0)).rgb * weights3[i];
				}
			} else {
				for (int i = 1; i < 3; ++i) {
					result += texture(brightness, v2fTexCoord + vec2(0.0, texelSize.y * offsets3[i])).rgb * weights3[i];
					result += texture(brightness, v2fTexCoord - vec2(0.0, texelSize.y * offsets3[i])).rgb * weights3[i];
				}
			}
			break;
		case 2: // 17x17 tap - offsets5 & weights5
			result = texture(brightness, v2fTexCoord).rgb * weights5[0];

			if (pConsts.direction == 0) {
				for (int i = 1; i < 5; ++i) {
					result += texture(brightness, v2fTexCoord + vec2(texelSize.x * offsets5[i], 0.0)).rgb * weights5[i];
					result += texture(brightness, v2fTexCoord - vec2(texelSize.x * offsets5[i], 0.0)).rgb * weights5[i];
				}
			} else {
				for (int i = 1; i < 5; ++i) {
					result += texture(brightness, v2fTexCoord + vec2(0.0, texelSize.y * offsets5[i])).rgb * weights5[i];
					result += texture(brightness, v2fTexCoord - vec2(0.0, texelSize.y * offsets5[i])).rgb * weights5[i];
				}
			}
			break;
		case 3: // 25z25 tap - offsets7 & weights7
			result = texture(brightness, v2fTexCoord).rgb * weights7[0];

			if (pConsts.direction == 0) {
				for (int i = 1; i < 7; ++i) {
					result += texture(brightness, v2fTexCoord + vec2(texelSize.x * offsets7[i], 0.0)).rgb * weights7[i];
					result += texture(brightness, v2fTexCoord - vec2(texelSize.x * offsets7[i], 0.0)).rgb * weights7[i];
				}
			} else {
				for (int i = 1; i < 7; ++i) {
					result += texture(brightness, v2fTexCoord + vec2(0.0, texelSize.y * offsets7[i])).rgb * weights7[i];
					result += texture(brightness, v2fTexCoord - vec2(0.0, texelSize.y * offsets7[i])).rgb * weights7[i];
				}
			}
			break;
		case 4: // 41x41 tap - offsets11 & weights11
			result = texture(brightness, v2fTexCoord).rgb * weights11[0];

			if (pConsts.direction == 0) {
				for (int i = 1; i < 11; ++i) {
					result += texture(brightness, v2fTexCoord + vec2(texelSize.x * offsets11[i], 0.0)).rgb * weights11[i];
					result += texture(brightness, v2fTexCoord - vec2(texelSize.x * offsets11[i], 0.0)).rgb * weights11[i];
				}
			} else {
				for (int i = 1; i < 11; ++i) {
					result += texture(brightness, v2fTexCoord + vec2(0.0, texelSize.y * offsets11[i])).rgb * weights11[i];
					result += texture(brightness, v2fTexCoord - vec2(0.0, texelSize.y * offsets11[i])).rgb * weights11[i];
				}
			}
			break;
	}

	oBlur = vec4(result, 1.0);
}