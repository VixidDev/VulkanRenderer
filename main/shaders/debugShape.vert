#version 450

layout(location = 0) in vec3 iPosition;				// rate: per vertex
layout(location = 1) in vec3 instanceTranslation;	// rate: per instance
layout(location = 2) in float instanceScale;		// rate: per instance
layout(location = 3) in vec4 instanceColour;		// rate: per instance

layout(set = 0, binding = 0) uniform MVP {
	mat4 projection;
	mat4 view;
	vec4 camPos;
} mvp;

layout(location = 0) out vec4 v2fColour;

void main() {
	v2fColour = instanceColour;

	vec3 pos = iPosition * instanceScale + instanceTranslation;
	gl_Position = mvp.projection * mvp.view * vec4(pos, 1.0);
}