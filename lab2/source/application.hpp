#pragma once

#include <vector>
#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "graphics_internal.hpp"

struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec3 normal;
};

// --- Структуры для света и материалов (выровнены по std140/std430) ---
struct alignas(16) Material {
    glm::vec4 albedo = glm::vec4(1.0f, 0.5f, 0.3f, 1.0f);
    glm::vec4 specularColor = glm::vec4(1.0f);
    float shininess = 32.0f;
    float _pad[3];
};

struct alignas(16) DirectionalLight {
    glm::vec4 direction = glm::vec4(-0.5f, -1.0f, -0.5f, 0.0f);
    glm::vec4 color = glm::vec4(1.0f);
};

struct alignas(16) PointLight {
    glm::vec4 position = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f);
    glm::vec4 color = glm::vec4(1.0f);
};

struct alignas(16) SpotLight {
    glm::vec4 position = glm::vec4(0.0f, 2.0f, 0.0f, 1.0f);
    glm::vec4 direction = glm::vec4(0.0f, -1.0f, 0.0f, 0.0f);
    glm::vec4 color = glm::vec4(1.0f);
    float innerCutoff = 0.9f;  // cos(25 deg)
    float outerCutoff = 0.8f;  // cos(35 deg)
    float _pad[2];
};

struct alignas(16) SceneUBO {
    glm::mat4 viewProj;
    glm::vec4 cameraPos;
    glm::vec4 ambientColor = glm::vec4(0.1f, 0.1f, 0.1f, 1.0f);
    DirectionalLight dirLight;
    int pointCount = 0;
    int spotCount = 0;
    float _pad[2];
};

struct alignas(16) ModelUBO {
    glm::mat4 model;
    Material material;
};

namespace application {

    bool initialize();
    void shutdown();
    void update(double time);
    void render(const graphics::internal::FrameData& frameData);

} // namespace application