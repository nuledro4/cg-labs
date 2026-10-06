#define _CRT_SECURE_NO_WARNINGS

#include "application.hpp"
#include <cmath>
#include <cstring>
#include <string>
#include <stdexcept>
#include <iostream>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "imgui.h"


namespace {

    struct Camera {
        glm::vec3 position = glm::vec3(0.0f, 1.0f, 4.0f);
        glm::vec3 target = glm::vec3(0.0f);
        glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
        float fov = 45.0f;
        float nearPlane = 0.1f;
        float farPlane = 100.0f;
    };

    struct ObjectState {
        glm::vec3 position = glm::vec3(0.0f);
        glm::vec3 rotation = glm::vec3(0.0f);
        glm::vec3 scale = glm::vec3(1.0f);
        glm::vec4 baseColor = glm::vec4(1.0f);

        bool  animate = false;
        float orbitRadius = 2.5f;
        float orbitHeight = 0.0f;
        float orbitSpeed = 1.0f;
        float orbitAngle = 0.0f;
    };

    struct UniformBufferObject {
        glm::mat4 mvp;
        glm::vec4 baseColor;
    };

    struct AppState {
        std::vector<Vertex>   vertices;
        std::vector<uint32_t> indices;

        VkBuffer      vertexBuffer = VK_NULL_HANDLE;
        VmaAllocation vertexAllocation = VK_NULL_HANDLE;
        VkBuffer      indexBuffer = VK_NULL_HANDLE;
        VmaAllocation indexAllocation = VK_NULL_HANDLE;

        VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
        VkPipelineLayout      pipelineLayout = VK_NULL_HANDLE;
        VkPipeline            graphicsPipeline = VK_NULL_HANDLE;
        VkDescriptorPool      descriptorPool = VK_NULL_HANDLE;

        static constexpr uint32_t OBJECT_COUNT = 3;
        std::array<VkBuffer, OBJECT_COUNT> uniformBuffers{};
        std::array<VmaAllocation, OBJECT_COUNT> uniformAllocations{};
        std::array<void*, OBJECT_COUNT> uniformMapped{};
        std::array<VkDescriptorSet, OBJECT_COUNT> descriptorSets{};

        std::array<ObjectState, OBJECT_COUNT> objects;
        Camera camera;

        bool  usePerspective = true;
        bool  isAnimating = false;
        float animationSpeed = 1.0f;
        double lastTime = 0.0;
    };

    AppState state;

    void generateSphere(std::vector<Vertex>& outVertices, std::vector<uint32_t>& outIndices, int stacks, int slices) {
        outVertices.clear();
        outIndices.clear();
        const float PI = 3.14159265359f;

        for (int i = 0; i <= stacks; ++i) {
            float phi = PI * float(i) / float(stacks);
            for (int j = 0; j <= slices; ++j) {
                float theta = 2.0f * PI * float(j) / float(slices);
                float x = std::sin(phi) * std::cos(theta);
                float y = std::cos(phi);
                float z = std::sin(phi) * std::sin(theta);

                Vertex v{};
                v.pos = glm::vec3(x, y, z);
                v.normal = glm::vec3(x, y, z);
                v.color = glm::vec3((x + 1.0f) * 0.5f, (y + 1.0f) * 0.5f, (z + 1.0f) * 0.5f);
                outVertices.push_back(v);
            }
        }

        for (int i = 0; i < stacks; ++i) {
            for (int j = 0; j < slices; ++j) {
                uint32_t first = uint32_t(i * (slices + 1) + j);
                uint32_t second = first + uint32_t(slices + 1);
                outIndices.push_back(first);  outIndices.push_back(second); outIndices.push_back(first + 1);
                outIndices.push_back(second); outIndices.push_back(second + 1); outIndices.push_back(first + 1);
            }
        }
    }


    void createSphereBuffers() {
        VmaAllocator allocator = graphics::internal::context.allocator;

        VkDeviceSize vSize = sizeof(Vertex) * state.vertices.size();
        VkBufferCreateInfo vInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        vInfo.size = vSize; vInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT; vInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo vAlloc{};
        vAlloc.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
        vmaCreateBuffer(allocator, &vInfo, &vAlloc, &state.vertexBuffer, &state.vertexAllocation, nullptr);
        void* vData = nullptr;
        vmaMapMemory(allocator, state.vertexAllocation, &vData);
        std::memcpy(vData, state.vertices.data(), size_t(vSize));
        vmaUnmapMemory(allocator, state.vertexAllocation);

        VkDeviceSize iSize = sizeof(uint32_t) * state.indices.size();
        VkBufferCreateInfo iInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        iInfo.size = iSize; iInfo.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT; iInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo iAlloc{};
        iAlloc.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
        vmaCreateBuffer(allocator, &iInfo, &iAlloc, &state.indexBuffer, &state.indexAllocation, nullptr);
        void* iData = nullptr;
        vmaMapMemory(allocator, state.indexAllocation, &iData);
        std::memcpy(iData, state.indices.data(), size_t(iSize));
        vmaUnmapMemory(allocator, state.indexAllocation);
    }

    void createDescriptorSetLayout() {
        VkDevice device = graphics::internal::context.device;
        VkDescriptorSetLayoutBinding uboBinding{};
        uboBinding.binding = 0;
        uboBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboBinding.descriptorCount = 1;
        uboBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo info{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        info.bindingCount = 1; info.pBindings = &uboBinding;
        vkCreateDescriptorSetLayout(device, &info, nullptr, &state.descriptorSetLayout);
    }

    void createDescriptorPool() {
        VkDevice device = graphics::internal::context.device;
        VkDescriptorPoolSize size{};
        size.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        size.descriptorCount = AppState::OBJECT_COUNT;

        VkDescriptorPoolCreateInfo info{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        info.poolSizeCount = 1; info.pPoolSizes = &size; info.maxSets = AppState::OBJECT_COUNT;
        vkCreateDescriptorPool(device, &info, nullptr, &state.descriptorPool);
    }

    void createDescriptorSets() {
        VkDevice device = graphics::internal::context.device;
        VmaAllocator allocator = graphics::internal::context.allocator;

        std::array<VkDescriptorSetLayout, AppState::OBJECT_COUNT> layouts;
        layouts.fill(state.descriptorSetLayout);

        VkDescriptorSetAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        allocInfo.descriptorPool = state.descriptorPool;
        allocInfo.descriptorSetCount = AppState::OBJECT_COUNT;
        allocInfo.pSetLayouts = layouts.data();
        vkAllocateDescriptorSets(device, &allocInfo, state.descriptorSets.data());

        for (uint32_t i = 0; i < AppState::OBJECT_COUNT; ++i) {
            VkDeviceSize size = sizeof(UniformBufferObject);
            VkBufferCreateInfo bufInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
            bufInfo.size = size; bufInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT; bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            VmaAllocationCreateInfo allocCreate{};
            allocCreate.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
            allocCreate.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

            VmaAllocationInfo allocResult{};
            vmaCreateBuffer(allocator, &bufInfo, &allocCreate, &state.uniformBuffers[i], &state.uniformAllocations[i], &allocResult);
            state.uniformMapped[i] = allocResult.pMappedData;

            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = state.uniformBuffers[i]; bufferInfo.offset = 0; bufferInfo.range = size;

            VkWriteDescriptorSet write{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            write.dstSet = state.descriptorSets[i]; write.dstBinding = 0; write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            write.descriptorCount = 1; write.pBufferInfo = &bufferInfo;
            vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
        }
    }

    void createGraphicsPipeline() {
        VkDevice device = graphics::internal::context.device;
        VkRenderPass renderPass = graphics::internal::context.render_pass;
        VkExtent2D extent = graphics::internal::context.swapchain_extent;

        VkPipelineLayoutCreateInfo plInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        plInfo.setLayoutCount = 1; plInfo.pSetLayouts = &state.descriptorSetLayout;
        vkCreatePipelineLayout(device, &plInfo, nullptr, &state.pipelineLayout);

        auto readFile = [](const std::string& fileName) {
            std::vector<std::string> candidates = {
                fileName,                                    
                "../" + fileName,                            
                "shaders/" + fileName,                       
                "../../shaders/" + fileName,
                
                "C:/Users/user/Downloads/vulkan-starter-app-master/shaders/" + fileName,
            };

            std::cout << "[shader] looking for: " << fileName << "\n";
            for (const auto& p : candidates) {
                FILE* f = nullptr;
                fopen_s(&f, p.c_str(), "rb");
                if (f) {
                    std::fseek(f, 0, SEEK_END);
                    size_t sz = std::ftell(f);
                    std::fseek(f, 0, SEEK_SET);
                    std::vector<char> buf(sz);
                    std::fread(buf.data(), 1, sz, f);
                    std::fclose(f);
                    std::cout << "[shader]   OK: " << p << " (" << sz << " bytes)\n";
                    return buf;
                }
                else {
                    std::cout << "[shader]   miss: " << p << "\n";
                }
            }
            throw std::runtime_error("Cannot open shader: " + fileName);
            };
        auto vertCode = readFile("simple.vert.spv");
        auto fragCode = readFile("simple.frag.spv");

        VkShaderModule vertModule, fragModule;
        VkShaderModuleCreateInfo ci{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
        ci.codeSize = vertCode.size(); ci.pCode = reinterpret_cast<const uint32_t*>(vertCode.data());
        vkCreateShaderModule(device, &ci, nullptr, &vertModule);
        ci.codeSize = fragCode.size(); ci.pCode = reinterpret_cast<const uint32_t*>(fragCode.data());
        vkCreateShaderModule(device, &ci, nullptr, &fragModule);

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT; stages[0].module = vertModule; stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module = fragModule; stages[1].pName = "main";

        VkVertexInputBindingDescription binding{};
        binding.binding = 0; binding.stride = sizeof(Vertex); binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        std::array<VkVertexInputAttributeDescription, 3> attrs{};
        attrs[0] = { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos) };
        attrs[1] = { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color) };
        attrs[2] = { 2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) };

        VkPipelineVertexInputStateCreateInfo vi{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
        vi.vertexBindingDescriptionCount = 1; vi.pVertexBindingDescriptions = &binding;
        vi.vertexAttributeDescriptionCount = uint32_t(attrs.size()); vi.pVertexAttributeDescriptions = attrs.data();

        VkPipelineInputAssemblyStateCreateInfo ia{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkViewport viewport{}; viewport.width = float(extent.width); viewport.height = float(extent.height); viewport.maxDepth = 1.0f;
        VkRect2D scissor{}; scissor.extent = extent;
        VkPipelineViewportStateCreateInfo vp{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
        vp.viewportCount = 1; vp.pViewports = &viewport; vp.scissorCount = 1; vp.pScissors = &scissor;

        VkPipelineRasterizationStateCreateInfo rs{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
        rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_BACK_BIT; rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo ms{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo ds{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
        ds.depthTestEnable = VK_FALSE; 
        ds.depthWriteEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState cba{};
        cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo cb{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
        cb.attachmentCount = 1; cb.pAttachments = &cba;

        VkGraphicsPipelineCreateInfo pi{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
        pi.stageCount = 2; pi.pStages = stages; pi.pVertexInputState = &vi; pi.pInputAssemblyState = &ia;
        pi.pViewportState = &vp; pi.pRasterizationState = &rs; pi.pMultisampleState = &ms;
        pi.pDepthStencilState = &ds; pi.pColorBlendState = &cb; pi.layout = state.pipelineLayout;
        pi.renderPass = renderPass; pi.subpass = 0;

        vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pi, nullptr, &state.graphicsPipeline);
        vkDestroyShaderModule(device, vertModule, nullptr);
        vkDestroyShaderModule(device, fragModule, nullptr);
    }

    void updateUniformBuffer(uint32_t objectIndex) {
        const ObjectState& obj = state.objects[objectIndex];
        glm::mat4 model(1.0f);
        model = glm::translate(model, obj.position);
        model = glm::rotate(model, glm::radians(obj.rotation.x), glm::vec3(1, 0, 0));
        model = glm::rotate(model, glm::radians(obj.rotation.y), glm::vec3(0, 1, 0));
        model = glm::rotate(model, glm::radians(obj.rotation.z), glm::vec3(0, 0, 1));
        model = glm::scale(model, obj.scale);

        glm::mat4 view = glm::lookAt(state.camera.position, state.camera.target, state.camera.up);

        float aspect = float(graphics::internal::context.swapchain_extent.width) / float(graphics::internal::context.swapchain_extent.height);
        glm::mat4 proj;
        if (state.usePerspective) {
            proj = glm::perspective(glm::radians(state.camera.fov), aspect, state.camera.nearPlane, state.camera.farPlane);
            proj[1][1] *= -1.0f;
        }
        else {
            float h = 5.0f; float w = h * aspect;
            proj = glm::ortho(-w, w, -h, h, state.camera.nearPlane, state.camera.farPlane);
            proj[1][1] *= -1.0f;
        }

        UniformBufferObject ubo{};
        ubo.mvp = proj * view * model;
        ubo.baseColor = obj.baseColor;
        std::memcpy(state.uniformMapped[objectIndex], &ubo, sizeof(ubo));
    }

} 

namespace application {

    bool initialize() {
        try {
            generateSphere(state.vertices, state.indices, 9, 9);
            createSphereBuffers();
            createDescriptorSetLayout();
            createDescriptorPool();
            createDescriptorSets();
            createGraphicsPipeline();

            state.objects[0].position = glm::vec3(-2.0f, 0.0f, 0.0f); state.objects[0].baseColor = glm::vec4(1.0f, 0.4f, 0.4f, 1.0f); state.objects[0].animate = true;
            state.objects[1].position = glm::vec3(0.0f, 0.0f, 0.0f); state.objects[1].baseColor = glm::vec4(0.4f, 1.0f, 0.4f, 1.0f);
            state.objects[2].position = glm::vec3(2.0f, 0.0f, 0.0f); state.objects[2].baseColor = glm::vec4(0.4f, 0.4f, 1.0f, 1.0f); state.objects[2].scale = glm::vec3(0.6f);

            state.lastTime = 0.0;
            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "Init error: " << e.what() << std::endl;
            return false;
        }
    }

    void shutdown() {
        VkDevice device = graphics::internal::context.device;
        VmaAllocator allocator = graphics::internal::context.allocator;

        vkDeviceWaitIdle(device);

        if (state.graphicsPipeline) {
            vkDestroyPipeline(device, state.graphicsPipeline, nullptr);
            state.graphicsPipeline = VK_NULL_HANDLE;
        }
        if (state.pipelineLayout) {
            vkDestroyPipelineLayout(device, state.pipelineLayout, nullptr);
            state.pipelineLayout = VK_NULL_HANDLE;
        }
        if (state.descriptorPool) {
            vkDestroyDescriptorPool(device, state.descriptorPool, nullptr);
            state.descriptorPool = VK_NULL_HANDLE;
        }
        if (state.descriptorSetLayout) {
            vkDestroyDescriptorSetLayout(device, state.descriptorSetLayout, nullptr);
            state.descriptorSetLayout = VK_NULL_HANDLE;
        }

        for (uint32_t i = 0; i < AppState::OBJECT_COUNT; ++i) {
            if (state.uniformBuffers[i]) {
                vmaDestroyBuffer(allocator, state.uniformBuffers[i], state.uniformAllocations[i]);
                state.uniformBuffers[i] = VK_NULL_HANDLE;
                state.uniformAllocations[i] = VK_NULL_HANDLE;
                state.uniformMapped[i] = nullptr;
            }
        }
        if (state.vertexBuffer) {
            vmaDestroyBuffer(allocator, state.vertexBuffer, state.vertexAllocation);
            state.vertexBuffer = VK_NULL_HANDLE;
            state.vertexAllocation = VK_NULL_HANDLE;
        }
        if (state.indexBuffer) {
            vmaDestroyBuffer(allocator, state.indexBuffer, state.indexAllocation);
            state.indexBuffer = VK_NULL_HANDLE;
            state.indexAllocation = VK_NULL_HANDLE;
        }
    }

    void update(double time) {
        float deltaTime = float(time - state.lastTime);
        state.lastTime = time;

        ImGui::Begin("Lab #1 - Sphere (Variant 5)");
        ImGui::Text("Projection");
        if (ImGui::RadioButton("Perspective", state.usePerspective)) state.usePerspective = true;
        ImGui::SameLine();
        if (ImGui::RadioButton("Orthographic", !state.usePerspective)) state.usePerspective = false;
        ImGui::Separator();

        for (uint32_t i = 0; i < AppState::OBJECT_COUNT; ++i) {
            ImGui::PushID(int(i));
            if (ImGui::CollapsingHeader(("Object " + std::to_string(i)).c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::DragFloat3("Position", glm::value_ptr(state.objects[i].position), 0.05f);
                ImGui::DragFloat3("Rotation", glm::value_ptr(state.objects[i].rotation), 1.0f);
                ImGui::DragFloat3("Scale", glm::value_ptr(state.objects[i].scale), 0.05f, 0.05f, 10.0f);
                ImGui::ColorEdit4("Base Color", glm::value_ptr(state.objects[i].baseColor));
                ImGui::Checkbox("Animate", &state.objects[i].animate);
                if (state.objects[i].animate) {
                    ImGui::SliderFloat("Orbit Radius", &state.objects[i].orbitRadius, 0.5f, 8.0f);
                    ImGui::SliderFloat("Orbit Height", &state.objects[i].orbitHeight, -3.0f, 3.0f);
                    ImGui::SliderFloat("Orbit Speed", &state.objects[i].orbitSpeed, 0.1f, 5.0f);
                }
            }
            ImGui::PopID();
        }

        ImGui::Separator();
        ImGui::Checkbox("Global Play", &state.isAnimating);
        ImGui::SliderFloat("Global Speed", &state.animationSpeed, 0.1f, 5.0f);
        ImGui::End();

        for (uint32_t i = 0; i < AppState::OBJECT_COUNT; ++i) {
            if (state.objects[i].animate && state.isAnimating) {
                state.objects[i].orbitAngle += deltaTime * state.animationSpeed * state.objects[i].orbitSpeed;
                float a = state.objects[i].orbitAngle;
                state.objects[i].position.x = state.objects[i].orbitRadius * std::cos(a);
                state.objects[i].position.z = state.objects[i].orbitRadius * std::sin(a);
                state.objects[i].position.y = state.objects[i].orbitHeight;
                state.objects[i].rotation.y = glm::degrees(a);
            }
            updateUniformBuffer(i);
        }
    }

    void render(const graphics::internal::FrameData& frameData) {
        VkCommandBuffer cmd = frameData.command_buffer;
        auto& ctx = graphics::internal::context;

        VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        vkBeginCommandBuffer(cmd, &beginInfo);

        VkClearValue clearValues[2]{};
        clearValues[0].color = { { 0.10f, 0.10f, 0.12f, 1.0f } };
        clearValues[1].depthStencil = { 1.0f, 0 };

        VkRenderPassBeginInfo rpInfo{ VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
        rpInfo.renderPass = ctx.render_pass;
        rpInfo.framebuffer = frameData.framebuffer;
        rpInfo.renderArea.offset = { 0, 0 };
        rpInfo.renderArea.extent = ctx.swapchain_extent;
        rpInfo.clearValueCount = 2;        
        rpInfo.pClearValues = clearValues;

        vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, state.graphicsPipeline);

        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &state.vertexBuffer, &offset);
        vkCmdBindIndexBuffer(cmd, state.indexBuffer, 0, VK_INDEX_TYPE_UINT32);

        for (uint32_t i = 0; i < AppState::OBJECT_COUNT; ++i) {
            vkCmdBindDescriptorSets(cmd,
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                state.pipelineLayout,
                0, 1,
                &state.descriptorSets[i],
                0, nullptr);
            vkCmdDrawIndexed(cmd,
                static_cast<uint32_t>(state.indices.size()),
                1, 0, 0, 0);
        }

        vkCmdEndRenderPass(cmd);
        vkEndCommandBuffer(cmd);
    }

} 