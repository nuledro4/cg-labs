#define _CRT_SECURE_NO_WARNINGS

#include "application.hpp"
#include <cmath>
#include <cstring>
#include <string>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "imgui.h"

namespace {

    constexpr uint32_t MAX_POINT_LIGHTS = 100;
    constexpr uint32_t MAX_SPOT_LIGHTS = 100;
    constexpr uint32_t MAX_OBJECTS = 5;

    struct Camera {
        glm::vec3 position = glm::vec3(0.0f, 1.0f, 5.0f);
        glm::vec3 target = glm::vec3(0.0f);
        glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
        float fov = 45.0f;
        float yaw = -90.0f;
        float pitch = 0.0f;
    };

    struct ObjectState {
        glm::vec3 position = glm::vec3(0.0f);
        glm::vec3 rotation = glm::vec3(0.0f);
        glm::vec3 scale = glm::vec3(1.0f);
        Material material;
        bool animate = false;
        float animSpeed = 1.0f;
        float orbitRadius = 2.5f;  // <-- Радиус круга
        float orbitHeight = 0.0f;  // <-- Высота над плоскостью
        float orbitAngle = 0.0f;
    };

    struct AppState {
        std::vector<Vertex> sphereVerts;
        std::vector<uint32_t> sphereIndices;
        VkBuffer sphereVB = VK_NULL_HANDLE; VmaAllocation sphereVBA = VK_NULL_HANDLE;
        VkBuffer sphereIB = VK_NULL_HANDLE; VmaAllocation sphereIBA = VK_NULL_HANDLE;

        std::vector<Vertex> cubeVerts;
        std::vector<uint32_t> cubeIndices;
        VkBuffer cubeVB = VK_NULL_HANDLE; VmaAllocation cubeVBA = VK_NULL_HANDLE;
        VkBuffer cubeIB = VK_NULL_HANDLE; VmaAllocation cubeIBA = VK_NULL_HANDLE;

        Camera camera;
        SceneUBO sceneData;
        std::vector<PointLight> pointLights;
        std::vector<SpotLight> spotLights;
        std::vector<ObjectState> objects;

        VkBuffer sceneUboBuf = VK_NULL_HANDLE; VmaAllocation sceneUboAlloc = VK_NULL_HANDLE; void* sceneUboMapped = nullptr;
        VkBuffer pointSsboBuf = VK_NULL_HANDLE; VmaAllocation pointSsboAlloc = VK_NULL_HANDLE; void* pointSsboMapped = nullptr;
        VkBuffer spotSsboBuf = VK_NULL_HANDLE; VmaAllocation spotSsboAlloc = VK_NULL_HANDLE; void* spotSsboMapped = nullptr;

        std::array<VkBuffer, MAX_OBJECTS> modelUboBufs{};
        std::array<VmaAllocation, MAX_OBJECTS> modelUboAllocs{};
        std::array<void*, MAX_OBJECTS> modelUboMapped{};

        VkDescriptorSetLayout sceneSetLayout = VK_NULL_HANDLE;
        VkDescriptorSetLayout modelSetLayout = VK_NULL_HANDLE;
        VkDescriptorPool descPool = VK_NULL_HANDLE;
        VkDescriptorSet sceneDescSet = VK_NULL_HANDLE;
        std::array<VkDescriptorSet, MAX_OBJECTS> modelDescSets{};

        VkPipelineLayout mainPipelineLayout = VK_NULL_HANDLE;
        VkPipeline mainPipeline = VK_NULL_HANDLE;
        VkPipelineLayout lightPipelineLayout = VK_NULL_HANDLE;
        VkPipeline lightPipeline = VK_NULL_HANDLE;

        bool showLights = true;
        double lastTime = 0.0;
    };

    AppState state;

    void generateSphere(std::vector<Vertex>& v, std::vector<uint32_t>& i, int stacks, int slices) {
        v.clear(); i.clear();
        const float PI = 3.14159265359f;
        for (int y = 0; y <= stacks; ++y) {
            for (int x = 0; x <= slices; ++x) {
                float xSeg = (float)x / slices;
                float ySeg = (float)y / stacks;
                float xPos = std::cos(xSeg * 2.0f * PI) * std::sin(ySeg * PI);
                float yPos = std::cos(ySeg * PI);
                float zPos = std::sin(xSeg * 2.0f * PI) * std::sin(ySeg * PI);

                Vertex vert{};
                vert.pos = glm::vec3(xPos, yPos, zPos);
                vert.normal = glm::vec3(xPos, yPos, zPos);
                vert.color = glm::vec3((xPos + 1) / 2, (yPos + 1) / 2, (zPos + 1) / 2);
                v.push_back(vert);
            }
        }
        for (int y = 0; y < stacks; ++y) {
            for (int x = 0; x < slices; ++x) {
                uint32_t first = y * (slices + 1) + x;
                uint32_t second = first + slices + 1;
                i.push_back(first); i.push_back(second); i.push_back(first + 1);
                i.push_back(second); i.push_back(second + 1); i.push_back(first + 1);
            }
        }
    }

    void generateCube(std::vector<Vertex>& v, std::vector<uint32_t>& i) {
        v.clear(); i.clear();
        float s = 0.5f;
        glm::vec3 positions[] = {
            {-s,-s,-s}, { s,-s,-s}, { s, s,-s}, {-s, s,-s},
            {-s,-s, s}, { s,-s, s}, { s, s, s}, {-s, s, s}
        };
        glm::vec3 normals[] = { {0,0,-1}, {0,0,1}, {0,1,0}, {0,-1,0}, {-1,0,0}, {1,0,0} };
        uint32_t indices[] = {
            0,2,1, 0,3,2, 4,5,6, 4,6,7, 3,6,2, 3,7,6, 0,1,5, 0,5,4, 0,4,7, 0,7,3, 1,2,6, 1,6,5
        };
        for (int f = 0; f < 6; ++f) {
            for (int j = 0; j < 4; ++j) {
                Vertex vert{};
                vert.pos = positions[f * 4 + j];
                vert.normal = normals[f];
                vert.color = glm::vec3(1.0f);
                v.push_back(vert);
            }
        }
        for (uint32_t idx : indices) i.push_back(idx);
    }

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VmaMemoryUsage memUsage, VkBuffer& buf, VmaAllocation& alloc, void** mapped = nullptr) {
        VkBufferCreateInfo info{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        info.size = size; info.usage = usage; info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = memUsage;
        if (mapped) allocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

        VmaAllocationInfo resultInfo{};
        vmaCreateBuffer(graphics::internal::context.allocator, &info, &allocInfo, &buf, &alloc, &resultInfo);
        if (mapped) *mapped = resultInfo.pMappedData;
    }

    void uploadGeometry(const std::vector<Vertex>& v, const std::vector<uint32_t>& i, VkBuffer& vb, VmaAllocation& vba, VkBuffer& ib, VmaAllocation& iba) {
        VkDeviceSize vSize = sizeof(Vertex) * v.size();
        VkDeviceSize iSize = sizeof(uint32_t) * i.size();
        createBuffer(vSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, vb, vba);
        createBuffer(iSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, ib, iba);

        void* data = nullptr;
        vmaMapMemory(graphics::internal::context.allocator, vba, &data);
        memcpy(data, v.data(), vSize); vmaUnmapMemory(graphics::internal::context.allocator, vba);
        vmaMapMemory(graphics::internal::context.allocator, iba, &data);
        memcpy(data, i.data(), iSize); vmaUnmapMemory(graphics::internal::context.allocator, iba);
    }

    // Улучшенный поиск шейдеров с выводом в консоль
    VkShaderModule createShaderModule(const std::string& filename) {
        // Ищем и в корне, и в папке shaders (с учетом CWD)
        std::vector<std::string> candidates = {
            filename,                                   // "shader.vert.spv" (рядом с exe)
            "shaders/" + filename,                      // "shaders/shader.vert.spv" (если CWD корень)
            "../shaders/" + filename,                   // "../shaders/shader.vert.spv" (если CWD build-debug)
            "../../shaders/" + filename,                // "../../shaders/shader.vert.spv" (если CWD build-debug/Debug)
            "C:/Users/user/Downloads/vulkan-starter-app-master/shaders/" + filename // Абсолютный путь
        };

        std::ifstream file;
        std::string loadedPath;
        for (const auto& p : candidates) {
            file.open(p, std::ios::binary | std::ios::ate);
            if (file.is_open()) {
                loadedPath = p;
                break;
            }
        }

        if (!file.is_open()) {
            std::string msg = "\n!!! ШЕЙДЕР НЕ НАЙДЕН: " + filename + "\nИскал тут:\n";
            for (const auto& p : candidates) msg += " - " + p + "\n";
            std::cerr << msg << std::endl;
            throw std::runtime_error(msg);
        }

        std::cout << "[Shader] OK: " << loadedPath << std::endl;

        size_t sz = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> buf(sz);
        file.read(buf.data(), sz);
        file.close();

        VkShaderModuleCreateInfo ci{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
        ci.codeSize = sz;
        ci.pCode = reinterpret_cast<const uint32_t*>(buf.data());
        VkShaderModule mod;
        vkCreateShaderModule(graphics::internal::context.device, &ci, nullptr, &mod);
        return mod;
    }

    VkPipeline createPipeline(const std::string& vertPath, const std::string& fragPath, VkPipelineLayout layout, bool useNormals) {
        VkDevice device = graphics::internal::context.device;
        VkShaderModule vertMod = createShaderModule(vertPath);
        VkShaderModule fragMod = createShaderModule(fragPath);

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vertMod, "main" };
        stages[1] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fragMod, "main" };

        VkVertexInputBindingDescription binding{ 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX };
        std::vector<VkVertexInputAttributeDescription> attrs = {
            { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos) },
            { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color) }
        };
        if (useNormals) attrs.push_back({ 2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) });

        VkPipelineVertexInputStateCreateInfo vi{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
        vi.vertexBindingDescriptionCount = 1; vi.pVertexBindingDescriptions = &binding;
        vi.vertexAttributeDescriptionCount = (uint32_t)attrs.size(); vi.pVertexAttributeDescriptions = attrs.data();

        VkPipelineInputAssemblyStateCreateInfo ia{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkExtent2D ext = graphics::internal::context.swapchain_extent;
        VkViewport vp{ 0, 0, (float)ext.width, (float)ext.height, 0, 1 };
        VkRect2D sc{ {0,0}, ext };
        VkPipelineViewportStateCreateInfo vps{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, nullptr, 0, 1, &vp, 1, &sc };

        VkPipelineRasterizationStateCreateInfo rs{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
        rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_BACK_BIT; rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo ms{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo ds{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
        ds.depthTestEnable = VK_FALSE; ds.depthWriteEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState cba{};
        cba.colorWriteMask = 0xF;
        VkPipelineColorBlendStateCreateInfo cbs{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, nullptr, 0, VK_FALSE, VK_LOGIC_OP_COPY, 1, &cba };

        VkGraphicsPipelineCreateInfo pi{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
        pi.stageCount = 2; pi.pStages = stages;
        pi.pVertexInputState = &vi; pi.pInputAssemblyState = &ia;
        pi.pViewportState = &vps; pi.pRasterizationState = &rs; pi.pMultisampleState = &ms;
        pi.pDepthStencilState = &ds; pi.pColorBlendState = &cbs;
        pi.layout = layout; pi.renderPass = graphics::internal::context.render_pass;

        VkPipeline pipe;
        vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pi, nullptr, &pipe);
        vkDestroyShaderModule(device, vertMod, nullptr); vkDestroyShaderModule(device, fragMod, nullptr);
        return pipe;
    }

} // namespace

namespace application {

    bool initialize() {
        try {
            std::cout << "[Init] Loading geometry..." << std::endl;
            generateSphere(state.sphereVerts, state.sphereIndices, 9, 9);
            uploadGeometry(state.sphereVerts, state.sphereIndices, state.sphereVB, state.sphereVBA, state.sphereIB, state.sphereIBA);
            generateCube(state.cubeVerts, state.cubeIndices);
            uploadGeometry(state.cubeVerts, state.cubeIndices, state.cubeVB, state.cubeVBA, state.cubeIB, state.cubeIBA);

            state.objects.resize(3);
            state.objects[0].position = glm::vec3(-2.0f, 0.0f, 0.0f);
            state.objects[0].material.albedo = glm::vec4(0.8f, 0.2f, 0.2f, 1.0f);
            state.objects[1].position = glm::vec3(0.0f, 0.0f, 0.0f);
            state.objects[1].material.albedo = glm::vec4(0.2f, 0.8f, 0.2f, 1.0f);
            state.objects[2].position = glm::vec3(2.0f, 0.0f, 0.0f);
            state.objects[2].material.albedo = glm::vec4(0.2f, 0.2f, 0.8f, 1.0f);

            state.pointLights.push_back({ glm::vec4(0, 1.5f, 0, 1), glm::vec4(1, 0, 0, 1) });
            state.spotLights.push_back({ glm::vec4(0, 3, 0, 1), glm::vec4(0, -1, 0, 0), glm::vec4(1, 1, 1, 1), 0.9f, 0.7f });

            createBuffer(sizeof(SceneUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, state.sceneUboBuf, state.sceneUboAlloc, &state.sceneUboMapped);
            createBuffer(sizeof(PointLight) * MAX_POINT_LIGHTS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, state.pointSsboBuf, state.pointSsboAlloc, &state.pointSsboMapped);
            createBuffer(sizeof(SpotLight) * MAX_SPOT_LIGHTS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, state.spotSsboBuf, state.spotSsboAlloc, &state.spotSsboMapped);

            for (int i = 0; i < MAX_OBJECTS; ++i) {
                createBuffer(sizeof(ModelUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, state.modelUboBufs[i], state.modelUboAllocs[i], &state.modelUboMapped[i]);
            }

            VkDevice dev = graphics::internal::context.device;

            VkDescriptorSetLayoutBinding sBindings[3]{};
            sBindings[0] = { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL_GRAPHICS };
            sBindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT };
            sBindings[2] = { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT };
            VkDescriptorSetLayoutCreateInfo slci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 3, sBindings };
            vkCreateDescriptorSetLayout(dev, &slci, nullptr, &state.sceneSetLayout);

            VkDescriptorSetLayoutBinding mBinding{ 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL_GRAPHICS };
            VkDescriptorSetLayoutCreateInfo mlci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 1, &mBinding };
            vkCreateDescriptorSetLayout(dev, &mlci, nullptr, &state.modelSetLayout);

            VkDescriptorPoolSize poolSizes[2] = {
                { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, MAX_OBJECTS + 10 },
                { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 }
            };
            VkDescriptorPoolCreateInfo dpci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, MAX_OBJECTS + 10, 2, poolSizes };
            vkCreateDescriptorPool(dev, &dpci, nullptr, &state.descPool);

            VkDescriptorSetAllocateInfo dsai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, state.descPool, 1, &state.sceneSetLayout };
            vkAllocateDescriptorSets(dev, &dsai, &state.sceneDescSet);

            VkDescriptorBufferInfo sInfo[3] = {
                { state.sceneUboBuf, 0, VK_WHOLE_SIZE },
                { state.pointSsboBuf, 0, VK_WHOLE_SIZE },
                { state.spotSsboBuf, 0, VK_WHOLE_SIZE }
            };
            VkWriteDescriptorSet sw[3]{};
            for (int i = 0; i < 3; ++i) {
                sw[i] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, state.sceneDescSet, (uint32_t)i, 0, 1, (VkDescriptorType)(i == 0 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER), nullptr, &sInfo[i], nullptr };
            }
            vkUpdateDescriptorSets(dev, 3, sw, 0, nullptr);

            for (int i = 0; i < MAX_OBJECTS; ++i) {
                VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, state.descPool, 1, &state.modelSetLayout };
                vkAllocateDescriptorSets(dev, &ai, &state.modelDescSets[i]);
                VkDescriptorBufferInfo bi{ state.modelUboBufs[i], 0, VK_WHOLE_SIZE };
                VkWriteDescriptorSet w{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, state.modelDescSets[i], 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &bi, nullptr };
                vkUpdateDescriptorSets(dev, 1, &w, 0, nullptr);
            }

            VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            VkDescriptorSetLayout sets[] = { state.sceneSetLayout, state.modelSetLayout };
            plci.setLayoutCount = 2; plci.pSetLayouts = sets;
            vkCreatePipelineLayout(dev, &plci, nullptr, &state.mainPipelineLayout);
            vkCreatePipelineLayout(dev, &plci, nullptr, &state.lightPipelineLayout);

            std::cout << "[Init] Creating Main Pipeline..." << std::endl;
            state.mainPipeline = createPipeline("shader.vert.spv", "shader.frag.spv", state.mainPipelineLayout, true);

            std::cout << "[Init] Creating Light Pipeline..." << std::endl;
            state.lightPipeline = createPipeline("light.vert.spv", "light.frag.spv", state.lightPipelineLayout, false);

            std::cout << "[Init] Success!" << std::endl;
            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "Init error: " << e.what() << "\n";
            shutdown(); // Очищаем память!
            return false;
        }
    }

    void shutdown() {
        VkDevice dev = graphics::internal::context.device;
        VmaAllocator alloc = graphics::internal::context.allocator;
        if (dev == VK_NULL_HANDLE) return;

        vkDeviceWaitIdle(dev);

        if (state.mainPipeline) vkDestroyPipeline(dev, state.mainPipeline, nullptr);
        if (state.lightPipeline) vkDestroyPipeline(dev, state.lightPipeline, nullptr);
        if (state.mainPipelineLayout) vkDestroyPipelineLayout(dev, state.mainPipelineLayout, nullptr);
        if (state.lightPipelineLayout) vkDestroyPipelineLayout(dev, state.lightPipelineLayout, nullptr);
        if (state.descPool) vkDestroyDescriptorPool(dev, state.descPool, nullptr);
        if (state.sceneSetLayout) vkDestroyDescriptorSetLayout(dev, state.sceneSetLayout, nullptr);
        if (state.modelSetLayout) vkDestroyDescriptorSetLayout(dev, state.modelSetLayout, nullptr);

        if (state.sceneUboBuf) vmaDestroyBuffer(alloc, state.sceneUboBuf, state.sceneUboAlloc);
        if (state.pointSsboBuf) vmaDestroyBuffer(alloc, state.pointSsboBuf, state.pointSsboAlloc);
        if (state.spotSsboBuf) vmaDestroyBuffer(alloc, state.spotSsboBuf, state.spotSsboAlloc);
        for (int i = 0; i < MAX_OBJECTS; ++i) {
            if (state.modelUboBufs[i]) vmaDestroyBuffer(alloc, state.modelUboBufs[i], state.modelUboAllocs[i]);
        }
        if (state.sphereVB) vmaDestroyBuffer(alloc, state.sphereVB, state.sphereVBA);
        if (state.sphereIB) vmaDestroyBuffer(alloc, state.sphereIB, state.sphereIBA);
        if (state.cubeVB) vmaDestroyBuffer(alloc, state.cubeVB, state.cubeVBA);
        if (state.cubeIB) vmaDestroyBuffer(alloc, state.cubeIB, state.cubeIBA);
    }

    void update(double time) {
        float dt = (float)(time - state.lastTime);
        state.lastTime = time;

        for (int i = 0; i < 3; ++i) {
            if (state.objects[i].animate) {
                state.objects[i].orbitAngle += dt * state.objects[i].animSpeed;

                float a = state.objects[i].orbitAngle;

                state.objects[i].position.x = state.objects[i].orbitRadius * std::cos(a);
                state.objects[i].position.z = state.objects[i].orbitRadius * std::sin(a);
                state.objects[i].position.y = state.objects[i].orbitHeight;

                state.objects[i].rotation.y = glm::degrees(a);
            }
        }

        ImGui::Begin("Lab");

        if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderFloat("Yaw", &state.camera.yaw, -180.0f, 180.0f);
            ImGui::SliderFloat("Pitch", &state.camera.pitch, -89.0f, 89.0f);
            ImGui::SliderFloat("FOV", &state.camera.fov, 10.0f, 120.0f);

            glm::vec3 dir;
            dir.x = cos(glm::radians(state.camera.yaw)) * cos(glm::radians(state.camera.pitch));
            dir.y = sin(glm::radians(state.camera.pitch));
            dir.z = sin(glm::radians(state.camera.yaw)) * cos(glm::radians(state.camera.pitch));

            state.camera.position = state.camera.target - glm::normalize(dir) * 6.0f;
        }

        if (ImGui::CollapsingHeader("Directional Light")) {
            ImGui::DragFloat3("Dir", glm::value_ptr(state.sceneData.dirLight.direction), 0.1f);
            ImGui::ColorEdit4("Color", glm::value_ptr(state.sceneData.dirLight.color));
        }
        ImGui::ColorEdit4("Ambient", glm::value_ptr(state.sceneData.ambientColor));

        if (ImGui::CollapsingHeader("Point Lights")) {
            if (ImGui::Button("Add Point Light")) state.pointLights.push_back({});
            for (int i = 0; i < state.pointLights.size(); ++i) {
                ImGui::PushID(i);
                if (ImGui::Button("X")) { state.pointLights.erase(state.pointLights.begin() + i); ImGui::PopID(); break; }
                ImGui::SameLine(); ImGui::Text("Light %d", i);
                ImGui::DragFloat3("Pos", glm::value_ptr(state.pointLights[i].position), 0.1f);
                ImGui::ColorEdit4("Col", glm::value_ptr(state.pointLights[i].color));
                ImGui::PopID();
            }
        }

        if (ImGui::CollapsingHeader("Spot Lights")) {
            if (ImGui::Button("Add Spot Light")) state.spotLights.push_back({});
            for (int i = 0; i < state.spotLights.size(); ++i) {
                ImGui::PushID(i + 1000);
                if (ImGui::Button("X")) { state.spotLights.erase(state.spotLights.begin() + i); ImGui::PopID(); break; }
                ImGui::SameLine(); ImGui::Text("Spot %d", i);
                ImGui::DragFloat3("Pos", glm::value_ptr(state.spotLights[i].position), 0.1f);
                ImGui::DragFloat3("Dir", glm::value_ptr(state.spotLights[i].direction), 0.1f);
                ImGui::ColorEdit4("Col", glm::value_ptr(state.spotLights[i].color));
                ImGui::SliderFloat("Inner", &state.spotLights[i].innerCutoff, 0.0f, 1.0f);
                ImGui::SliderFloat("Outer", &state.spotLights[i].outerCutoff, 0.0f, 1.0f);
                ImGui::PopID();
            }
        }

        if (ImGui::CollapsingHeader("Object Transforms", ImGuiTreeNodeFlags_DefaultOpen)) {
            for (int i = 0; i < 3; ++i) {
                ImGui::PushID(i + 3000);
                ImGui::Text("Sphere %d", i);
                ImGui::DragFloat3("Position", glm::value_ptr(state.objects[i].position), 0.05f);
                ImGui::DragFloat3("Rotation", glm::value_ptr(state.objects[i].rotation), 1.0f);
                ImGui::DragFloat3("Scale", glm::value_ptr(state.objects[i].scale), 0.05f, 0.05f, 10.0f);
                ImGui::Checkbox("Animate Orbit", &state.objects[i].animate);
                if (state.objects[i].animate) {
                    ImGui::SliderFloat("Speed", &state.objects[i].animSpeed, 0.1f, 5.0f);
                    ImGui::SliderFloat("Orbit Radius", &state.objects[i].orbitRadius, 0.5f, 5.0f);
                    ImGui::SliderFloat("Orbit Height", &state.objects[i].orbitHeight, -3.0f, 3.0f);
                }
                
                ImGui::PopID();
            }
        }

        if (ImGui::CollapsingHeader("Object Materials")) {
            for (int i = 0; i < 3; ++i) {
                ImGui::PushID(i + 2000);
                ImGui::Text("Sphere %d", i);
                ImGui::ColorEdit4("Albedo", glm::value_ptr(state.objects[i].material.albedo));
                ImGui::ColorEdit4("Spec", glm::value_ptr(state.objects[i].material.specularColor));
                ImGui::SliderFloat("Shininess", &state.objects[i].material.shininess, 1.0f, 256.0f);
                ImGui::PopID();
            }
        }
        ImGui::Checkbox("Show Light Sources", &state.showLights);
        ImGui::End();

        // --- Матрицы ---
        float aspect = (float)graphics::internal::context.swapchain_extent.width / (float)graphics::internal::context.swapchain_extent.height;
        glm::mat4 view = glm::lookAt(state.camera.position, state.camera.target, state.camera.up);
        glm::mat4 proj = glm::perspective(glm::radians(state.camera.fov), aspect, 0.1f, 100.0f);
        proj[1][1] *= -1.0f;

        state.sceneData.viewProj = proj * view;
        state.sceneData.cameraPos = glm::vec4(state.camera.position, 1.0f);
        state.sceneData.pointCount = (int)state.pointLights.size();
        state.sceneData.spotCount = (int)state.spotLights.size();

        memcpy(state.sceneUboMapped, &state.sceneData, sizeof(SceneUBO));
        if (!state.pointLights.empty())
            memcpy(state.pointSsboMapped, state.pointLights.data(), state.pointLights.size() * sizeof(PointLight));
        if (!state.spotLights.empty())
            memcpy(state.spotSsboMapped, state.spotLights.data(), state.spotLights.size() * sizeof(SpotLight));

        for (int i = 0; i < MAX_OBJECTS; ++i) {
            ModelUBO m{};
            if (i < 3) {
                m.model = glm::translate(glm::mat4(1.0f), state.objects[i].position);
                m.model = glm::rotate(m.model, glm::radians(state.objects[i].rotation.x), glm::vec3(1, 0, 0));
                m.model = glm::rotate(m.model, glm::radians(state.objects[i].rotation.y), glm::vec3(0, 1, 0));
                m.model = glm::rotate(m.model, glm::radians(state.objects[i].rotation.z), glm::vec3(0, 0, 1));
                m.model = glm::scale(m.model, state.objects[i].scale);
            }
            else if (i == 3 && state.pointLights.size() > 0) {
                // Визуализация точечного света (кубик)
                m.model = glm::translate(glm::mat4(1.0f), glm::vec3(state.pointLights[0].position)) * glm::scale(glm::mat4(1.0f), glm::vec3(0.1f));
                m.material.albedo = state.pointLights[0].color;
            }
            else {
                m.model = glm::scale(glm::mat4(1.0f), glm::vec3(0.001f)); // Скрываем остальные
            }
            memcpy(state.modelUboMapped[i], &m, sizeof(ModelUBO));
        }
    }

    void render(const graphics::internal::FrameData& frameData) {
        VkCommandBuffer cmd = frameData.command_buffer;

        VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        vkBeginCommandBuffer(cmd, &bi);

        VkClearValue clear{};
        clear.color = { {0.05f, 0.05f, 0.08f, 1.0f} };
        VkRenderPassBeginInfo rp{ VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
        rp.renderPass = graphics::internal::context.render_pass;
        rp.framebuffer = frameData.framebuffer;
        rp.renderArea = { {0,0}, graphics::internal::context.swapchain_extent };
        rp.clearValueCount = 1; rp.pClearValues = &clear;
        vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

        VkDeviceSize offsets[] = { 0 };

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, state.mainPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, state.mainPipelineLayout, 0, 1, &state.sceneDescSet, 0, nullptr);

        vkCmdBindVertexBuffers(cmd, 0, 1, &state.sphereVB, offsets);
        vkCmdBindIndexBuffer(cmd, state.sphereIB, 0, VK_INDEX_TYPE_UINT32);

        for (int i = 0; i < 3; ++i) {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, state.mainPipelineLayout, 1, 1, &state.modelDescSets[i], 0, nullptr);
            vkCmdDrawIndexed(cmd, (uint32_t)state.sphereIndices.size(), 1, 0, 0, 0);
        }

        if (state.showLights && state.pointLights.size() > 0) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, state.lightPipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, state.lightPipelineLayout, 0, 1, &state.sceneDescSet, 0, nullptr);

            vkCmdBindVertexBuffers(cmd, 0, 1, &state.cubeVB, offsets);
            vkCmdBindIndexBuffer(cmd, state.cubeIB, 0, VK_INDEX_TYPE_UINT32);

            // Рисуем куб на месте первого точечного света
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, state.lightPipelineLayout, 1, 1, &state.modelDescSets[3], 0, nullptr);
            vkCmdDrawIndexed(cmd, (uint32_t)state.cubeIndices.size(), 1, 0, 0, 0);
        }

        vkCmdEndRenderPass(cmd);
        vkEndCommandBuffer(cmd);
    }

} // namespace application