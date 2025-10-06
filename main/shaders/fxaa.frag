#version 450

// Some platforms may need this explicitly enabled
// if not compiling using glslc
//#extension GL_GOOGLE_include_directive : enable

// FXAA defines for highest quality FXAA
#define FXAA_PC 1
#define FXAA_GLSL_130 1
#define FXX_QUALITY__PRESET 39

#include "Fxaa3_11.h"

layout(location = 0) in noperspective vec2 v2fTexCoord;

layout(set = 0, binding = 0) uniform sampler2D inputImage;

layout(location = 0) out vec4 oColour;

void main() {
	vec2 rcpFrame = 1.0 / textureSize(inputImage, 0);

	oColour = FxaaPixelShader(
		v2fTexCoord,		// FxaaFloat2 pos
		FxaaFloat4(0.0),	// FxaaFloat4 fxaaConsolePosPos
		inputImage,			// FxaaTex	  tex
		inputImage,			// FxaaTex    fxaaConsole360TexExpBiasNegOne
		inputImage,			// FxaaTex    fxaaConsole360TexExpBiasNegTwo
		rcpFrame,			// FxaaFloat2 fxaaQualityRcpFrame
		FxaaFloat4(0.0),	// FxaaFloat4 fxaaConsoleRcpFrameOpt
		FxaaFloat4(0.0),	// FxaaFloat4 fxaaConsoleRcpFrameOpt2
		FxaaFloat4(0.0),	// FxaaFloat4 fxaaConsole360RcpFrameOpt2
		1.0,				// FxaaFloat  fxaaQualitySubpix
		0.166,				// FxaaFloat  fxaaQualityEdgeThreshold
		0.0833,				// FxaaFloat  fxaaQualityEdgeThresholdMin
		0.0,				// FxaaFloat  fxaaConsoleEdgeSharpness
		0.0,				// FxaaFloat  fxaaConsoleEdgeThreshold
		0.0,				// FxaaFloat  fxaaConsoleEdgeThresholdMin
		FxaaFloat4(0.0)		// FxaaFloat4 fxaaConsole360ConstDir
	);
}