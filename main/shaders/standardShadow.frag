#version 450

layout(location = 0) in vec2 v2fTexCoord;

layout(set = 0, binding = 0) uniform ClipPlanes {
	float far;
	float near;
} planes;

layout(set = 1, binding = 0) uniform sampler2D uAlphaMask;

layout(push_constant) uniform PushConstant {
	layout(offset = 64) int projectionType; // 0 - Perspective, 1 - Orthographic
} pConsts;

layout(location = 0) out vec4 linearDepth;

layout(constant_id = 0) const int WRITE_TO_COLOUR = 0;

float lineariseDepth(float depth) {
	return (2.0 * planes.near * planes.far) / (planes.far + planes.near - (2.0 * depth - 1.0) * (planes.far - planes.near));
}

void main() {
	float alpha = texture(uAlphaMask, v2fTexCoord).a;
	if (alpha < 0.5) discard;

	// WRITE_TO_COLOUR will only be equal to 1 in debug mode since
	// in debug thats the only time there exists a render target for
	// linearDepth
	if (WRITE_TO_COLOUR == 1) {
		// Only perspective projected shadow maps need their depth linearising,
		// orthographic projected shadow maps already write depth linearly.
		if (pConsts.projectionType == 0) {
			float linearisedDepth = (lineariseDepth(gl_FragCoord.z) - planes.near) / (planes.far - planes.near);
			linearDepth = vec4(vec3(linearisedDepth), 1.0);
		} else {
			linearDepth = vec4(vec3(gl_FragCoord.z), 1.0);
		}
	}
}