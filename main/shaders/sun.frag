#version 450

layout(location = 0) in vec2 v2fTexCoord;

layout(set = 0, binding = 0) uniform MVP {
	mat4 projection;
	mat4 view;
	vec4 camPos;
} mvp;

layout(set = 1, binding = 0) uniform InverseMatrices {
	// just invView * invProj done on CPU instead, saves doing it in fragment shader
	mat4 invViewProj;
	mat4 invProj;
	mat4 invView;
} inverses;

layout(location = 0) out vec4 oColour;

void main() {
	vec4 clipPos = vec4(v2fTexCoord * 2.0 - 1.0, 0.0, 1.0);
	vec4 worldPos = inverses.invViewProj * clipPos;
	worldPos /= worldPos.w;

	vec3 viewDir = normalize(worldPos.xyz - mvp.camPos.xyz);
	float angleToSun = distance(viewDir, -vec3(0.34815531, -0.8703882, -0.34815531));
	vec3 intensity = smoothstep(0.03, 0.026, angleToSun) * vec3(1.0) * 50.0;

	if (intensity.rgb == vec3(0.0)) discard;

	oColour = vec4(intensity, 1.0);
}