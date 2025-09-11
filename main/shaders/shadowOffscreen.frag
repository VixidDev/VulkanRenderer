#version 450

layout(set = 1, binding = 0) uniform ClipPlanes {
	float far;
	float near;
} planes;

layout(location = 0) out vec4 linearDepth;

float lineariseDepth(float depth) {
	return (2.0f * planes.near * planes.far) / (planes.far + planes.near - (2.0f * depth - 1.0f) * (planes.far - planes.near));
}

void main() {
	linearDepth = vec4(vec3(lineariseDepth(gl_FragCoord.z) / 100.0f), 1.0f);
}