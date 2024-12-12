#version 450

layout(location = 0) in vec3 iPosition;
layout(location = 1) in vec2 iTexCoord;
layout(location = 2) in vec3 iNormal;

layout(set = 0, binding = 0) uniform UScene {
    mat4 camera;
    mat4 projection;
    mat4 projCam;
    vec4 camPos;
} uScene;

layout(location = 0) out vec2 v2fTexCoord;
layout(location = 1) out vec3 v2fNormal;
layout(location = 2) out vec3 v2fPosition; // World-coord position

void main(){
    v2fTexCoord = iTexCoord;
    v2fNormal = iNormal;
    v2fPosition = iPosition;

    gl_Position = uScene.projCam * vec4(iPosition, 1.0f);
}