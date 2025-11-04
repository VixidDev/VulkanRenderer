#version 450

layout(location = 0) in vec2 v2fTexCoord;

layout(binding = 0) uniform sampler2D resultImage;

layout(location = 0) out vec4 oColor;

void main() {
    vec2 size = textureSize(resultImage, 0);

    // Kernel sizes essentially results in reducing 
    // width resolution by 5 and width resolution by 3
    float kernelWidth  = 1.0 / (size.x / 5.0);
    float kernelHeight = 1.0 / (size.y / 3.0);

    vec2 sampleCoord = vec2(kernelWidth  * floor(v2fTexCoord.x / kernelWidth), 
                            kernelHeight * floor(v2fTexCoord.y / kernelHeight));
                            
    // Get size of half a pixel
    vec2 offset = 0.5 / size;

    // Clamp the sample coord to at least be half a pixel from either edge of the screen
    sampleCoord = clamp(sampleCoord, 0 + offset, 1 - offset);

    oColor = texture(resultImage, sampleCoord);
}