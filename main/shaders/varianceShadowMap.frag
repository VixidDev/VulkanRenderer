#version 450

// world space position for point lights and spot lights
// light clip space position for directional lights
layout(location = 0) in vec4 v2fPosition;
layout(location = 1) in vec2 v2fTexCoord;

layout(set = 0, binding = 0) uniform sampler2D uAlphaMask;

layout(location = 0) out vec2 oMoments;

layout(push_constant) uniform PushConstants {
	layout(offset = 128) vec3 lightPos;
	layout(offset = 144) float farPlane;
} pConsts;

layout(constant_id = 0) const int LIGHT_TYPE = 0;

void main() {
	float alpha = texture(uAlphaMask, v2fTexCoord).a;
	if (alpha < 0.5) discard;

	float linearDepth;
	float distToLight;
	switch (LIGHT_TYPE) {
		case 0: // Point lights
			distToLight = length(v2fPosition.xyz - pConsts.lightPos);
			linearDepth = distToLight / pConsts.farPlane;
			break;
		case 1: // Directional light
			linearDepth = v2fPosition.z;
			break;
		case 2: // Spot lights
			distToLight = length(v2fPosition.xyz - pConsts.lightPos);
			linearDepth = distToLight / pConsts.farPlane;
			break;
	}

	oMoments = vec2(linearDepth, linearDepth * linearDepth);
}