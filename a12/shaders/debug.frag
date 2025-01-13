#version 450

layout(location = 0) in vec2 v2fTexCoord;

layout(set = 1, binding = 0) uniform sampler2D uTexColor;
layout(set = 1, binding = 1) uniform sampler2D uMetalness;
layout(set = 1, binding = 2) uniform sampler2D uRoughness;

layout(set = 2, binding = 0) uniform Debug {
    int debug;
} debug;

layout(location = 0) out vec4 oColor;

float linearise_depth(float d) {
    return (2.0 * 0.1f * 100.0f) / (100.0f + 0.1f - (2 * d - 1.0f) * (100.0f - 0.1f));
}

void main() {
    // Mipmap Utilisation Visualisation
    switch(debug.debug) {
        case 2:
            float mipmapLevel = textureQueryLod(uTexColor, v2fTexCoord).x;

            // Colors for mipmaps
            // Colorblind friendly colors https://davidmathlogic.com/colorblind/
            vec3 colors[7];
            colors[0] = vec3(17.0, 119.0, 51.0) / 255;   // Murky green
            colors[1] = vec3(136.0, 34.0, 85.0) / 255;   // Purple
            colors[2] = vec3(0.0, 114.0, 178.0) / 255;   // Blue
            colors[3] = vec3(204.0, 121.0, 167.0) / 255; // Pink
            colors[4] = vec3(0.0, 158.0, 115.0) / 255;   // Turquoise
            colors[5] = vec3(213.0, 94.0, 0.0) / 255;    // Orange
            colors[6] = vec3(240.0, 228.0, 66.0) / 255;  // Yellow

            vec3 floorColor = colors[int(floor(mipmapLevel)) % 7];
            vec3 ceilColor = colors[int(ceil(mipmapLevel)) % 7];

            oColor = vec4(mix(floorColor, ceilColor, fract(mipmapLevel)), 1.0f);
            break;
        case 3:
            // Linearised Fragment Depth Visualisation
            oColor = vec4(vec3(linearise_depth(gl_FragCoord.z) / 100.0f), 1.0f);
            break;
        case 4:
            // Partial derivatives of per-fragment depth visualisation
            float depth = linearise_depth(gl_FragCoord.z);

            float dx = dFdx(depth) * 5;
            float dy = dFdy(depth) * 5;

            // float dx = abs(dFdx(gl_FragCoord.z)) * 200;
            // float dy = abs(dFdy(gl_FragCoord.z)) * 200;

            oColor = vec4(dx, dy, 0.0f, 1.0f);
            break;
        default:
            // Normal rendering (in this case the normal rendering pipeline should be used
            // but we put this here just in case something goes wrong somewhere)
            oColor = texture(uTexColor, v2fTexCoord).rgba;
    }

}
