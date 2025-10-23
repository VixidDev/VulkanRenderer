#version 450

layout(location = 0) in vec3 iPosition;
layout(location = 1) in vec2 iTexCoord;

// world space position for point and spot lights
// light clip space position for directional light
layout(location = 0) out vec4 v2fPosition;
layout(location = 1) out vec2 v2fTexCoord;

layout(push_constant) uniform PushConstants {
	mat4 lightViewProj;
	mat4 lightView;
} pConsts;

layout(constant_id = 0) const int LIGHT_TYPE = 0;

void main() {
	gl_Position = pConsts.lightViewProj * vec4(iPosition, 1.0);

	switch (LIGHT_TYPE) {
		case 0: // Point lights (cubemap)
			v2fPosition = vec4(iPosition, 1.0);
			break;
		case 1: // Directional light
			v2fPosition = gl_Position;
			break;
		case 2: // Spot lights
			v2fPosition = vec4(iPosition, 1.0);
			break;
	}

	v2fTexCoord = iTexCoord;	
}
