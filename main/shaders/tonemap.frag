#version 450

#define TONEMAP_FILMIC 0
#define TONEMAP_UNCHARTED 1
#define TONEMAP_ACES 2
#define TONEMAP_AGX 3
#define TONEMAP_KHRONOS_PBR 4

#define AGX_LOOK 0 // 0 - Default, 1 - Golden, 2 - Punchy

layout(location = 0) in vec2 v2fTexCoord;

layout(set = 0, binding = 0) uniform sampler2D inputImage;

layout(location = 0) out vec4 oColour;

layout(push_constant) uniform PushConstant {
	int tonemapType;
	float exposure;
} pConsts;

vec3 linearToSrgb(vec3 rgb) {
	vec3 low = rgb * 12.92;
	vec3 high = 1.055 * pow(rgb, vec3(1.0 / 2.4)) - 0.055;
	bvec3 cutoff = lessThan(rgb, vec3(0.0031308));

	return mix(high, low, cutoff);
}

// Filmic tonemapping operator by Jim Hejl and Richard Burgess-Dawson
// Optimised version that approximates the Digital Fusion Cineon mode,
// sRGB correction built in
// http://filmicworlds.com/blog/filmic-tonemapping-operators/
vec3 tonemapFilmic(vec3 rgb) {
	vec3 x = max(vec3(0.0), rgb - 0.004);
	return (x * (6.2 * x + 0.5)) / (x * (6.2 * x + 1.7) + 0.06);
}

vec3 tonemapUncharted2Impl(vec3 rgb) {
	const float A = 0.15;
	const float B = 0.50;
	const float C = 0.10;
	const float D = 0.20;
	const float E = 0.02;
	const float F = 0.30;
	return ((rgb * (A * rgb + C * B) + D * E) / (rgb * (A * rgb + B) + D * F)) - E / F;
}

// Tonemapping operator used in Uncharted 2 by John Hable.
// http://filmicworlds.com/blog/filmic-tonemapping-operators/
vec3 tonemapUncharted2(vec3 rgb) {
	const float exposureBias = 2.0;
	vec3 first = tonemapUncharted2Impl(exposureBias * rgb);
	vec3 whiteScale = 1.0 / tonemapUncharted2Impl(vec3(11.2));
	vec3 colour = first * whiteScale;
	// Could use linearToSrgb here, but original implementation
	// just uses pow(rgb, 1/2.2)
	return pow(colour, vec3(1.0 / 2.2));
}

const mat3 ACESInputMat = mat3(
    0.59719, 0.07600, 0.02840,
    0.35458, 0.90834, 0.13383,
    0.04823, 0.01566, 0.83777);

const mat3 ACESOutputMat = mat3(
     1.60475, -0.10208, -0.00327,
    -0.53108,  1.10813, -0.07276,
    -0.07367, -0.00605,  1.07602);

// ACES tonemapping approximation by Stephen Hill
// https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl
vec3 tonemapACES(vec3 rgb) {
	rgb = ACESInputMat * rgb;

	// Apply RRT and ODT
	vec3 a = rgb * (rgb + 0.0245786) - 0.000090537;
	vec3 b = rgb * (0.983729 * rgb + 0.4329510) + 0.238081;
	rgb = a / b;

	rgb = ACESOutputMat * rgb;
	return linearToSrgb(rgb);
}

vec3 AgXDefaultContrastApprox(vec3 rgb) {
	vec3 rgb2 = rgb * rgb;
	vec3 rgb4 = rgb2 * rgb2;
  
	return + 15.5     * rgb4 * rgb2
		   - 40.14    * rgb4 * rgb
           + 31.96    * rgb4
           - 6.868    * rgb2 * rgb
           + 0.4298   * rgb2
           + 0.1191   * rgb
           - 0.00232;
}

vec3 AgXLook(vec3 rgb) {
	// Default
	// (offset is never actually used but is included
	// for completeness)
	vec3 offset = vec3(0.0);
	vec3 slope = vec3(1.0);
	vec3 power = vec3(1.0);
	float saturation = 1.0;

#if (AGX_LOOK == 1) // Golden
	slope = vec3(1.0, 0.9, 0.5);
	power = vec3(0.8);
	saturation = 0.8;
#elif (AGX_LOOK == 2) // Punchy
	power = vec3(1.35);
	saturation = 1.4;
#endif

	// ASC CDL
	rgb = pow(rgb * slope + offset, power);

	const vec3 lw = vec3(0.2126, 0.7152, 0.0722);
	float luma = dot(rgb, lw);

	return luma + saturation * (rgb - luma);
}

const mat3 AgXMat = mat3(
    0.842479062253094,  0.0423282422610123, 0.0423756549057051,
    0.0784335999999992, 0.878468636469772,  0.0784336,
    0.0792237451477643, 0.0791661274605434, 0.879142973793104);

const mat3 AgXMatInv = mat3(
     1.19687900512017,   -0.0528968517574562, -0.0529716355144438,
    -0.0980208811401368,  1.15190312990417,   -0.0980434501171241,
    -0.0990297440797205, -0.0989611768448433,  1.15107367264116);

// Approximation of Troy Sobotka's AgX by Benjamin Wrensch
// https://iolite-engine.com/blog_posts/minimal_agx_implementation
vec3 tonemapAgX(vec3 rgb) {
	const float minEv = -12.47393;
	const float maxEv = 4.026069;

	// Input transform (inset)
	rgb = AgXMat * rgb;

	// Log2 space encoding
	rgb = clamp(log2(rgb), minEv, maxEv);
	rgb = (rgb - minEv) / (maxEv - minEv);
	
	// Apply sigmoid function approximation
	rgb = AgXDefaultContrastApprox(rgb);

#if 0
	//rgb = AgXLook(rgb); // Optional
#endif

	// Inverse input transform (outset)
	rgb = AgXMatInv * rgb;
	
	// Keep this commented out since we want to
	// stay in sRGB and not linearise
	//rgb = pow(rgb, vec3(2.2));

	return rgb;
}

// Khronos' Neutral PBR tonemapping
// https://github.com/KhronosGroup/ToneMapping/blob/main/PBR_Neutral/pbrNeutral.glsl
vec3 tonemapKhronosPBR(vec3 rgb) {
	const float startCompression = 0.8 - 0.04;
	const float desaturation = 0.15;

	float x = min(rgb.r, min(rgb.g, rgb.b));
	float offset = x < 0.08 ? x - 6.25 * x * x: 0.04;
	rgb -= offset;

	float peak = max(rgb.r, max(rgb.g, rgb.b));
	if (peak < startCompression) {
		return rgb;
	}

	const float d = 1.0 - startCompression;
	float newPeak = 1.0 - d * d / (peak + d - startCompression);
	rgb *= newPeak / peak;

	float g = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0);
	rgb = mix(rgb, vec3(newPeak), g);

	return linearToSrgb(rgb);
}

void main() {
	
	vec3 colour = texture(inputImage, v2fTexCoord).rgb;
	vec3 tonemapped = colour * pConsts.exposure;

	// Tonemap
	switch(pConsts.tonemapType) {
		case TONEMAP_FILMIC:
			tonemapped = tonemapFilmic(tonemapped);
			break;
		case TONEMAP_UNCHARTED:
			tonemapped = tonemapUncharted2(tonemapped);
			break;
		case TONEMAP_ACES:
			tonemapped = tonemapACES(tonemapped);
			break;
		case TONEMAP_AGX:
			tonemapped = tonemapAgX(tonemapped);
			break;
		case TONEMAP_KHRONOS_PBR:
			tonemapped = tonemapKhronosPBR(tonemapped);
			break;
	};

	// Get Luma
	float luma = dot(tonemapped, vec3(0.299, 0.587, 0.114));

	oColour = vec4(tonemapped, luma);
}