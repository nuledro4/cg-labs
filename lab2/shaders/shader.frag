#version 450
layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragColor;
layout(location = 0) out vec4 outColor;

struct PointLight { vec4 position; vec4 color; };
struct SpotLight { vec4 position; vec4 direction; vec4 color; float innerCutoff; float outerCutoff; vec2 _pad; };

layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 viewProj; vec4 cameraPos; vec4 ambientColor;
    vec4 dirLightDir; vec4 dirLightColor;
    int pointCount; int spotCount; vec2 _pad;
} scene;
layout(set = 0, binding = 1) readonly buffer PointLights { PointLight lights[]; } pointLights;
layout(set = 0, binding = 2) readonly buffer SpotLights { SpotLight lights[]; } spotLights;

layout(set = 1, binding = 0) uniform ModelUBO {
    mat4 model; vec4 albedo; vec4 specularColor;
    float shininess; vec3 _pad;
} ubo;

vec3 BlinnPhong(vec3 N, vec3 V, vec3 L, vec3 lightCol, float att) {
    vec3 H = normalize(V + L);
    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), ubo.shininess);
    vec3 diffuse = diff * lightCol * ubo.albedo.rgb;
    vec3 specular = spec * lightCol * ubo.specularColor.rgb;
    return (diffuse + specular) * att;
}

void main() {
    vec3 N = normalize(fragNormal);
    vec3 V = normalize(scene.cameraPos.xyz - fragPos);
    vec3 result = scene.ambientColor.rgb * ubo.albedo.rgb;

    // 1. Directional Light
    vec3 L = normalize(-scene.dirLightDir.xyz);
    vec3 H = normalize(V + L);
    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), ubo.shininess);
    result += (diff * ubo.albedo.rgb + spec * ubo.specularColor.rgb) * scene.dirLightColor.rgb;

    // 2. Point Lights (Обратный квадрат)
    for (int i = 0; i < scene.pointCount; ++i) {
        vec3 Lvec = pointLights.lights[i].position.xyz - fragPos;
        float dist = length(Lvec);
        float att = 1.0 / (dist * dist + 0.001); // Защита от деления на 0
        result += BlinnPhong(N, V, Lvec / dist, pointLights.lights[i].color.rgb, att);
    }

    // 3. Spot Lights (Мягкие края + Обратный квадрат)
    for (int i = 0; i < scene.spotCount; ++i) {
        vec3 Lvec = spotLights.lights[i].position.xyz - fragPos;
        float dist = length(Lvec);
        float att = 1.0 / (dist * dist + 0.001);
        vec3 L = Lvec / dist;
        
        float theta = dot(L, normalize(-spotLights.lights[i].direction.xyz));
        float epsilon = spotLights.lights[i].innerCutoff - spotLights.lights[i].outerCutoff;
        float intensity = clamp((theta - spotLights.lights[i].outerCutoff) / epsilon, 0.0, 1.0);
        
        if (intensity > 0.0) {
            result += BlinnPhong(N, V, L, spotLights.lights[i].color.rgb, att * intensity);
        }
    }

    outColor = vec4(result * fragColor, 1.0); // Умножаем на цвет вершин
}