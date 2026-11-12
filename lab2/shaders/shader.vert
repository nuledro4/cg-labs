#version 450
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 inNormal;

layout(location = 0) out vec3 fragPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragColor;

layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 viewProj;
    vec4 cameraPos;
    vec4 ambientColor;
    vec4 dirLightDir;
    vec4 dirLightColor;
    int pointCount;
    int spotCount;
} scene;

layout(set = 1, binding = 0) uniform ModelUBO {
    mat4 model;
    vec4 albedo;
    vec4 specularColor;
    float shininess;
    vec3 _pad;
} ubo;

void main() {
    vec4 worldPos = ubo.model * vec4(inPosition, 1.0);
    fragPos = worldPos.xyz;
    fragNormal = normalize(transpose(inverse(mat3(ubo.model))) * inNormal);
    fragColor = inColor; // Процедурный цвет
    gl_Position = scene.viewProj * worldPos;
}