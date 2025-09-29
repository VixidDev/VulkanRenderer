#version 450

layout(location = 0) in vec2 v2fTexCoord;

layout(set = 0, binding = 0) uniform sampler2D readImage1;
layout(set = 1, binding = 0) uniform sampler2D readImage2;

layout(location = 0) out vec4 oColour;

void main() {
	vec3 readImage1Col = texture(readImage1, v2fTexCoord).rgb;
	vec3 readImage2Col = texture(readImage2, v2fTexCoord).rgb;

	oColour = vec4(readImage1Col + readImage2Col, 1.0);
}