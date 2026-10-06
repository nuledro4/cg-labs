#pragma once

#include <vector>
#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "graphics_internal.hpp" // Содержит graphics::FrameData и graphics::internal::context

// Структура вершины
struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec3 normal;
};

namespace application {

    // Функции, которые вызывает main.cpp
    bool initialize();
    void shutdown();
    void update(double time);
    void render(const graphics::internal::FrameData& frameData);

} // namespace application