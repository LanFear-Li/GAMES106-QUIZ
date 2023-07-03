#pragma once

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE
#ifdef VK_USE_PLATFORM_ANDROID_KHR
#define TINYGLTF_ANDROID_LOAD_FROM_ASSETS
#endif

#include "tiny_gltf.h"
#include "vulkanexamplebase.h"

#define ENABLE_VALIDATION false

// Contains everything required to render a glTF model in Vulkan
class VulkanglTFModel
{
public:
    // The class requires some Vulkan objects, so it can create its own resources
    vks::VulkanDevice *vulkanDevice;
    VkQueue copyQueue;

    // Single vertex buffer for all primitives
    struct Vertices
    {
        VkBuffer buffer;
        VkDeviceMemory memory;
    } vertices;

    // Single index buffer for all primitives
    struct Indices
    {
        uint32_t count;
        VkBuffer buffer;
        VkDeviceMemory memory;
    } indices;

    // The following structures roughly represent the glTF scene structure
    // To keep things simple, they only contain those properties that are required for this sample
    struct Node;

    // A glTF material stores information in e.g. the texture that is attached to it and colors
    struct Material
    {
        uint32_t baseColorTextureIndex{};
        uint32_t metallicRoughnessTextureIndex{};
        uint32_t normalTextureIndex{};
        uint32_t emissiveTextureIndex{};
        uint32_t occlusionTextureIndex{};

        VkDescriptorSet descriptorSet{};
    };

    // Contains the texture for a single glTF image
    // Images may be reused by texture objects and are as such separated
    struct Image
    {
        vks::Texture2D texture;
        // We also store (and create) a descriptor set that's used to access this texture from the fragment shader
        VkDescriptorSet descriptorSet;
    };

    // A glTF texture stores a reference to the image and a sampler
    // In this sample, we are only interested in the image
    struct Texture
    {
        int32_t imageIndex;
    };

    // A primitive contains the data for a single draw call
    struct Primitive
    {
        uint32_t firstIndex;
        uint32_t indexCount;
        int32_t materialIndex;
    };

    // Contains the node's (optional) geometry and can be made up of an arbitrary number of primitives
    struct Mesh
    {
        std::vector<Primitive> primitives;
    };

    // A node represents an object in the glTF scene graph
    struct Node
    {
        Node *parent{};
        uint32_t index{};
        std::vector<Node *> children;
        Mesh mesh;
        glm::vec3 translation{};
        glm::vec3 scale{1.0f};
        glm::quat rotation{};
        int32_t skin = -1;
        glm::mat4 matrix;

        glm::mat4 getLocalMatrix();
    };

    // The vertex layout for the samples' model
    struct Vertex
    {
        glm::vec3 pos;
        glm::vec3 normal;
        glm::vec2 uv;
        glm::vec3 color;
        glm::vec4 tangent;
    };

    /*
		Skin structure
	*/

    struct Skin
    {
        std::string name;
        Node *skeletonRoot = nullptr;
        std::vector<glm::mat4> inverseBindMatrices;
        std::vector<Node *> joints;
        vks::Buffer ssbo;
        VkDescriptorSet descriptorSet;
    };

    /*
        Animation related structures
    */

    struct AnimationSampler
    {
        std::string interpolation;
        std::vector<float> inputs;
        std::vector<glm::vec4> outputsVec4;
    };

    struct AnimationChannel
    {
        std::string path;
        Node *node;
        uint32_t samplerIndex;
    };

    struct Skeleton
    {
        std::vector<glm::mat4> nodeTransform;
    } skeleton;

    struct Animation
    {
        std::string name;
        std::vector<AnimationSampler> samplers;
        std::vector<AnimationChannel> channels;
        float start = std::numeric_limits<float>::max();
        float end = std::numeric_limits<float>::min();
        float currentTime = 0.0f;
    };

    uint32_t activeAnimation = 0;

    /*
        Model data
    */
    std::vector<Image> images;
    std::vector<Texture> textures;
    std::vector<Material> materials;
    std::vector<Node *> nodes;
    std::vector<Skin> skins;
    std::vector<Animation> animations;

    ~VulkanglTFModel();

    void loadImages(tinygltf::Model &input);

    void loadTextures(tinygltf::Model &input);

    void loadMaterials(tinygltf::Model &input);

    Node *findNode(Node *parent, uint32_t index);

    Node *nodeFromIndex(uint32_t index);

    void loadSkins(tinygltf::Model &input);

    void loadAnimations(tinygltf::Model &input);

    void prepareAnimations(tinygltf::Model &input);

    void loadNode(const tinygltf::Node &inputNode, const tinygltf::Model &input, VulkanglTFModel::Node *parent,
                  uint32_t nodeIndex, std::vector<uint32_t> &indexBuffer,
                  std::vector<VulkanglTFModel::Vertex> &vertexBuffer);

    static glm::mat4 getNodeMatrix(VulkanglTFModel::Node *node);

    void updateJoints(VulkanglTFModel::Node *node);

    void updateNodeTransform(Node *node);

    void updateAnimation(float deltaTime);

    void drawNode(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, VulkanglTFModel::Node *node);

    void draw(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout);
};

class VulkanExample : public VulkanExampleBase
{
public:
    bool wireframe = false;

    VulkanglTFModel glTFModel;

    vks::Texture2D defaultOcclusionMap{};
    vks::Texture2D defaultEmissiveMap{};

    struct ShaderData
    {
        vks::Buffer buffer;
        struct Values
        {
            glm::mat4 projection;
            glm::mat4 model;
            glm::vec4 lightPos = glm::vec4(5.0f, 5.0f, 5.0f, 1.0f);
            glm::vec4 viewPos;
        } values;
    } shaderData;

    struct Pipelines
    {
        VkPipeline solid;
        VkPipeline wireframe = VK_NULL_HANDLE;
    } pipelines;

    VkPipelineLayout pipelineLayout{};
    VkDescriptorSet descriptorSet{};

    struct DescriptorSetLayouts
    {
        VkDescriptorSetLayout matrices;
        VkDescriptorSetLayout textures;
    } descriptorSetLayouts{};

    VulkanExample();

    ~VulkanExample() override;

    void loadglTFFile(const std::string &filename);

    void getEnabledFeatures() override;

    void buildCommandBuffers() override;

    void loadAssets();

    void loadDefaultTextureMap();

    void setupDescriptors();

    void preparePipelines();

    void prepareUniformBuffers();

    void updateUniformBuffers();

    void prepare() override;

    void render() override;

    void viewChanged() override;

    void OnUpdateUIOverlay(vks::UIOverlay *overlay) override;
};