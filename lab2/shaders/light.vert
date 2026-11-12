#version 450
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 fragColor;

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
    vec4 albedo; // Здесь мы передаем цвет лампочки
    vec4 specularColor;
    float shininess;
    vec3 _pad;
} ubo;

void main() {
    fragColor = ubo.albedo.rgb; // Берем цвет из материала
    gl_Position = scene.viewProj * ubo.model * vec4(inPosition, 1.0);
}