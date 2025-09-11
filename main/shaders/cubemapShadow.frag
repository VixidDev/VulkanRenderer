#version 450

layout(location = 0) in vec3 worldPos;

layout(push_constant) uniform PushConstant {
	layout(offset = 64) vec3 lightPos;
	layout(offset = 80) float farPlane;
} pConsts;

void main() {
	// For now light pos is hardcoded. Will be dynamic later
	float distToLight = length(worldPos - pConsts.lightPos);
	gl_FragDepth = distToLight / pConsts.farPlane;
}