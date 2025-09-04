#version 450

layout(location = 0) in vec3 worldPos;

layout(set = 1, binding = 0) uniform ClipPlanes {
	float far;
	float near;
	float bias;
} planes;

void main() {
	// For now light pos is hardcoded. Will be dynamic later
	float distToLight = length(worldPos - vec3(-0.2972f, 7.3100f, -11.9532f));
	gl_FragDepth = distToLight / planes.far;
}