#version 450

layout(location = 0) in vec3 iPosition;
layout(location = 1) in vec2 iTexCoord;
layout(location = 2) in vec4 iTBN;

layout(set = 0, binding = 0) uniform MVP {
	mat4 projection;
	mat4 view;
	vec4 camPos;
} mvp;

layout(set = 2, binding = 0) uniform Depth {
    mat4 depthMVP;
} depth;

layout(location = 0) out vec3 v2fPosition;
layout(location = 1) out vec2 v2fTexCoord;
layout(location = 2) out mat3 v2fTBN;

// Taken from mat3_cast in glm/gtc/quaternion.inl
mat3 quaternion_to_rot_matrix(vec4 q) {
    float qxx = q.x * q.x;
    float qyy = q.y * q.y;
    float qzz = q.z * q.z;
    float qxz = q.x * q.z;
    float qxy = q.x * q.y;
    float qyz = q.y * q.z;
    float qwx = q.w * q.x;
    float qwy = q.w * q.y;
    float qwz = q.w * q.z;

    return mat3(
        1.0f - 2.0f * (qyy + qzz), 2.0f * (qxy + qwz),        2.0f * (qxz - qwy),
        2.0f * (qxy - qwz),        1.0f - 2.0f * (qxx + qzz), 2.0f * (qyz + qwx),
        2.0f * (qxz + qwy),        2.0f * (qyz - qwx),        1.0f - 2.0f * (qxx + qyy)
    );
}

const mat4 biasMat = mat4( 
	0.5, 0.0, 0.0, 0.0,
	0.0, 0.5, 0.0, 0.0,
	0.0, 0.0, 1.0, 0.0,
	0.5, 0.5, 0.0, 1.0);

void main() {
    v2fPosition = iPosition;
    v2fTexCoord = iTexCoord;
    //v2fLightSpacePosition = (biasMat * depth.depthMVP) * vec4(iPosition, 1.0f);

    // Decode TBN

    // Remap smallest components to [-1/sqrt(2), 1/sqrt(2)]
    vec3 smallest = (iTBN.rgb * sqrt(2.0f) - (1/sqrt(2.0f)));
    // Using 1 = x^2 + y^2 + z^2 + w^2 identity
    float maxComponent = sqrt(1 - dot(smallest, smallest));
    // Get index of max component. Since 2 bits were mapped to [0.0f, 1.0f] as format is UNORM,
    // we need to multiply by 3 and round off, i.e. 
    //     0b00 (0)   = 0.0f, 0b01 (1)   = 0.33333f, 0b10 (2)   = 0.666666f, 0b11 (3)   = 1.0f
    //     round(* 3) = 0.0f, round(* 3) = 1.0f,     round(* 3) = 2.0f,      round(* 3) = 3.0f
    // Finally cast to int to be used as index
    int maxIndex = int(round(iTBN.a * 3));

    // Reconstruct quaternion
    vec4 quaternion = vec4(0.0f);
    int quatIndex = 0;
    for (int i = 0; i < 4; i++) {
        if (maxIndex != i) {
            quaternion[i] = smallest[quatIndex++];
        } else {
            quaternion[i] = maxComponent;
        }
    }

    v2fTBN = quaternion_to_rot_matrix(quaternion);

    gl_Position = mvp.projection * mvp.view * vec4(iPosition, 1.0f);
}