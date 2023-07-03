/*
* Vulkan Example - glTF scene loading and rendering
*
* Copyright (C) 2020-2022 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

/*
 * Shows how to load and display a simple scene from a glTF file
 * Note that this isn't a complete glTF loader and only basic functions are shown here
 * This means no complex materials, no animations, no skins, etc.
 * For details on how glTF 2.0 works, see the official spec at https://github.com/KhronosGroup/glTF/tree/master/specification/2.0
 *
 * Other samples will load models using a dedicated model loader with more features (see base/VulkanglTFModel.hpp)
 *
 * If you are looking for a complete glTF implementation, check out https://github.com/SaschaWillems/Vulkan-glTF-PBR/
 */

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
// This class is heavily simplified (compared to glTF's feature set) but retains the basic glTF structure
class VulkanglTFModel
{
public:
	// The class requires some Vulkan objects so it can create it's own resources
	vks::VulkanDevice* vulkanDevice;
	VkQueue copyQueue;

	// Single vertex buffer for all primitives
	struct {
		VkBuffer buffer;
		VkDeviceMemory memory;
	} vertices;

	// Single index buffer for all primitives
	struct {
		int count;
		VkBuffer buffer;
		VkDeviceMemory memory;
	} indices;

	// The following structures roughly represent the glTF scene structure
	// To keep things simple, they only contain those properties that are required for this sample

    struct Node;

    // A glTF material stores information in e.g. the texture that is attached to it and colors
    struct Material {
        uint32_t baseColorTextureIndex{};
        uint32_t metallicRoughnessTextureIndex{};
        uint32_t normalTextureIndex{};
        uint32_t emissiveTextureIndex{};
        uint32_t occlusionTextureIndex{};

        VkDescriptorSet descriptorSet{};
    };

    // Contains the texture for a single glTF image
    // Images may be reused by texture objects and are as such separated
    struct Image {
        vks::Texture2D texture;
        // We also store (and create) a descriptor set that's used to access this texture from the fragment shader
        VkDescriptorSet descriptorSet;
    };

    // A glTF texture stores a reference to the image and a sampler
    // In this sample, we are only interested in the image
    struct Texture {
        int32_t imageIndex;
    };

	// A primitive contains the data for a single draw call
	struct Primitive {
		uint32_t firstIndex;
		uint32_t indexCount;
		int32_t materialIndex;
	};

	// Contains the node's (optional) geometry and can be made up of an arbitrary number of primitives
	struct Mesh {
		std::vector<Primitive> primitives;
	};

	// A node represents an object in the glTF scene graph
	struct Node {
        Node *              parent{};
        uint32_t            index{};
        std::vector<Node *> children;
        Mesh                mesh;
        glm::vec3           translation{};
        glm::vec3           scale{1.0f};
        glm::quat           rotation{};
        int32_t             skin = -1;
        glm::mat4           matrix;

        glm::mat4 getLocalMatrix() {
            return glm::translate(glm::mat4(1.0f), translation) * glm::mat4(rotation) * glm::scale(glm::mat4(1.0f), scale) * matrix;
        }
	};

    // The vertex layout for the samples' model
    struct Vertex {
        glm::vec3 pos;
        glm::vec3 normal;
        glm::vec2 uv;
        glm::vec4 color;
        glm::vec4 tangent;
    };

    /*
		Skin structure
	*/

    struct Skin
    {
        std::string            name;
        Node *                 skeletonRoot = nullptr;
        std::vector<glm::mat4> inverseBindMatrices;
        std::vector<Node *>    joints;
        vks::Buffer            ssbo;
        VkDescriptorSet        descriptorSet;
    };

    /*
        Animation related structures
    */

    struct AnimationSampler
    {
        std::string            interpolation;
        std::vector<float>     inputs;
        std::vector<glm::vec4> outputsVec4;
    };

    struct AnimationChannel
    {
        std::string path;
        Node *      node;
        uint32_t    samplerIndex;
    };

    struct Skeleton
    {
        std::vector<glm::mat4> nodeTransform;
        vks::Buffer ssbo;
        VkDescriptorSet descriptorSet{};
    } skeleton;

    struct Animation
    {
        std::string                   name;
        std::vector<AnimationSampler> samplers;
        std::vector<AnimationChannel> channels;
        float                         start       = std::numeric_limits<float>::max();
        float                         end         = std::numeric_limits<float>::min();
        float                         currentTime = 0.0f;
    };

	/*
		Model data
	*/
	std::vector<Image> images;
	std::vector<Texture> textures;
	std::vector<Material> materials;
	std::vector<Node*> nodes;
    std::vector<Skin> skins;
    std::vector<Animation> animations;

    uint32_t activeAnimation = 0;

	~VulkanglTFModel() {
		// Release all Vulkan resources allocated for the model
		vkDestroyBuffer(vulkanDevice->logicalDevice, vertices.buffer, nullptr);
		vkFreeMemory(vulkanDevice->logicalDevice, vertices.memory, nullptr);
		vkDestroyBuffer(vulkanDevice->logicalDevice, indices.buffer, nullptr);
		vkFreeMemory(vulkanDevice->logicalDevice, indices.memory, nullptr);

		for (Image image : images) {
			vkDestroyImageView(vulkanDevice->logicalDevice, image.texture.view, nullptr);
			vkDestroyImage(vulkanDevice->logicalDevice, image.texture.image, nullptr);
			vkDestroySampler(vulkanDevice->logicalDevice, image.texture.sampler, nullptr);
			vkFreeMemory(vulkanDevice->logicalDevice, image.texture.deviceMemory, nullptr);
		}

        // Release all Vulkan resources allocated for the animation
        skeleton.ssbo.destroy();
	}

	/*
		glTF loading functions

		The following functions take a glTF input model loaded via tinyglTF and convert all required data into our own structure
	*/

	void loadImages(tinygltf::Model& input)
	{
		// Images can be stored inside the glTF (which is the case for the sample model), so instead of directly
		// loading them from disk, we fetch them from the glTF loader and upload the buffers
		images.resize(input.images.size());
		for (size_t i = 0; i < input.images.size(); i++) {
			tinygltf::Image& glTFImage = input.images[i];
			// Get the image data from the glTF loader
			unsigned char* buffer = nullptr;
			VkDeviceSize bufferSize = 0;
			bool deleteBuffer = false;
			// We convert RGB-only images to RGBA, as most devices don't support RGB-formats in Vulkan
			if (glTFImage.component == 3) {
				bufferSize = glTFImage.width * glTFImage.height * 4;
				buffer = new unsigned char[bufferSize];
				unsigned char* rgba = buffer;
				unsigned char* rgb = &glTFImage.image[0];
				for (size_t i = 0; i < glTFImage.width * glTFImage.height; ++i) {
					memcpy(rgba, rgb, sizeof(unsigned char) * 3);
					rgba += 4;
					rgb += 3;
				}
				deleteBuffer = true;
			}
			else {
				buffer = &glTFImage.image[0];
				bufferSize = glTFImage.image.size();
			}
			// Load texture from image buffer
			images[i].texture.fromBuffer(buffer, bufferSize, VK_FORMAT_R8G8B8A8_UNORM, glTFImage.width, glTFImage.height, vulkanDevice, copyQueue);
			if (deleteBuffer) {
				delete[] buffer;
			}
		}
	}

	void loadTextures(tinygltf::Model& input)
	{
		textures.resize(input.textures.size());
		for (size_t i = 0; i < input.textures.size(); i++) {
			textures[i].imageIndex = input.textures[i].source;
		}
	}

	void loadMaterials(tinygltf::Model& input)
	{
		materials.resize(input.materials.size());
		for (size_t i = 0; i < input.materials.size(); i++) {
			// We only read the most basic properties required for our sample
			tinygltf::Material glTFMaterial = input.materials[i];

            materials[i].baseColorTextureIndex = glTFMaterial.pbrMetallicRoughness.baseColorTexture.index;
            materials[i].metallicRoughnessTextureIndex = glTFMaterial.pbrMetallicRoughness.metallicRoughnessTexture.index;
            materials[i].normalTextureIndex = glTFMaterial.normalTexture.index;
            materials[i].occlusionTextureIndex = glTFMaterial.occlusionTexture.index;
            materials[i].emissiveTextureIndex = glTFMaterial.emissiveTexture.index;
		}
	}

    // Helper functions for locating glTF nodes

    Node *findNode(Node *parent, uint32_t index) {
        Node *nodeFound = nullptr;
        if (parent->index == index)
        {
            return parent;
        }
        for (auto &child : parent->children)
        {
            nodeFound = findNode(child, index);
            if (nodeFound)
            {
                break;
            }
        }
        return nodeFound;
    }

    Node *nodeFromIndex(uint32_t index) {
        Node *nodeFound = nullptr;
        for (auto &node : nodes)
        {
            nodeFound = findNode(node, index);
            if (nodeFound)
            {
                break;
            }
        }
        return nodeFound;
    }

    // POI: Load the skins from the glTF model
    void loadSkins(tinygltf::Model &input) {
        skins.resize(input.skins.size());

        for (size_t i = 0; i < input.skins.size(); i++)
        {
            tinygltf::Skin glTFSkin = input.skins[i];

            skins[i].name = glTFSkin.name;
            // Find the root node of the skeleton
            skins[i].skeletonRoot = nodeFromIndex(glTFSkin.skeleton);

            // Find joint nodes
            for (int jointIndex : glTFSkin.joints)
            {
                Node *node = nodeFromIndex(jointIndex);
                if (node)
                {
                    skins[i].joints.push_back(node);
                }
            }

            // Get the inverse bind matrices from the buffer associated to this skin
            if (glTFSkin.inverseBindMatrices > -1)
            {
                const tinygltf::Accessor &  accessor   = input.accessors[glTFSkin.inverseBindMatrices];
                const tinygltf::BufferView &bufferView = input.bufferViews[accessor.bufferView];
                const tinygltf::Buffer &    buffer     = input.buffers[bufferView.buffer];
                skins[i].inverseBindMatrices.resize(accessor.count);
                memcpy(skins[i].inverseBindMatrices.data(), &buffer.data[accessor.byteOffset + bufferView.byteOffset], accessor.count * sizeof(glm::mat4));

                // Store inverse bind matrices for this skin in a shader storage buffer object
                // To keep this sample simple, we create a host visible shader storage buffer
                VK_CHECK_RESULT(vulkanDevice->createBuffer(
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        &skins[i].ssbo,
                        sizeof(glm::mat4) * skins[i].inverseBindMatrices.size(),
                        skins[i].inverseBindMatrices.data()));
                VK_CHECK_RESULT(skins[i].ssbo.map());
            }
        }
    }

    // POI: Load the animations from the glTF model
    void loadAnimations(tinygltf::Model &input) {
        animations.resize(input.animations.size());

        for (size_t i = 0; i < input.animations.size(); i++) {
            tinygltf::Animation glTFAnimation = input.animations[i];
            animations[i].name                = glTFAnimation.name;

            // Samplers
            animations[i].samplers.resize(glTFAnimation.samplers.size());
            for (size_t j = 0; j < glTFAnimation.samplers.size(); j++)
            {
                tinygltf::AnimationSampler glTFSampler = glTFAnimation.samplers[j];
                AnimationSampler &         dstSampler  = animations[i].samplers[j];
                dstSampler.interpolation               = glTFSampler.interpolation;

                // Read sampler keyframe input time values
                {
                    const tinygltf::Accessor &  accessor   = input.accessors[glTFSampler.input];
                    const tinygltf::BufferView &bufferView = input.bufferViews[accessor.bufferView];
                    const tinygltf::Buffer &    buffer     = input.buffers[bufferView.buffer];
                    const void *                dataPtr    = &buffer.data[accessor.byteOffset + bufferView.byteOffset];
                    const float *               buf        = static_cast<const float *>(dataPtr);
                    for (size_t index = 0; index < accessor.count; index++)
                    {
                        dstSampler.inputs.push_back(buf[index]);
                    }
                    // Adjust animation's start and end times
                    for (auto input : animations[i].samplers[j].inputs)
                    {
                        if (input < animations[i].start)
                        {
                            animations[i].start = input;
                        };
                        if (input > animations[i].end)
                        {
                            animations[i].end = input;
                        }
                    }
                }

                // Read sampler keyframe output translate/rotate/scale values
                {
                    const tinygltf::Accessor &  accessor   = input.accessors[glTFSampler.output];
                    const tinygltf::BufferView &bufferView = input.bufferViews[accessor.bufferView];
                    const tinygltf::Buffer &    buffer     = input.buffers[bufferView.buffer];
                    const void *                dataPtr    = &buffer.data[accessor.byteOffset + bufferView.byteOffset];
                    switch (accessor.type)
                    {
                        case TINYGLTF_TYPE_VEC3: {
                            const glm::vec3 *buf = static_cast<const glm::vec3 *>(dataPtr);
                            for (size_t index = 0; index < accessor.count; index++)
                            {
                                dstSampler.outputsVec4.push_back(glm::vec4(buf[index], 0.0f));
                            }
                            break;
                        }
                        case TINYGLTF_TYPE_VEC4: {
                            const glm::vec4 *buf = static_cast<const glm::vec4 *>(dataPtr);
                            for (size_t index = 0; index < accessor.count; index++)
                            {
                                dstSampler.outputsVec4.push_back(buf[index]);
                            }
                            break;
                        }
                        default: {
                            std::cout << "unknown type" << std::endl;
                            break;
                        }
                    }
                }
            }

            // Channels
            animations[i].channels.resize(glTFAnimation.channels.size());
            for (size_t j = 0; j < glTFAnimation.channels.size(); j++) {
                tinygltf::AnimationChannel glTFChannel = glTFAnimation.channels[j];
                AnimationChannel &         dstChannel  = animations[i].channels[j];
                dstChannel.path                        = glTFChannel.target_path;
                dstChannel.samplerIndex                = glTFChannel.sampler;
                dstChannel.node                        = nodeFromIndex(glTFChannel.target_node);
            }
        }
    }

    void prepareAnimations(tinygltf::Model &input) {
        skeleton.nodeTransform.resize(input.nodes.size(), glm::mat4(1.f));

        VK_CHECK_RESULT(vulkanDevice->createBuffer(
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                &skeleton.ssbo,
                sizeof(glm::mat4) * skeleton.nodeTransform.size(),skeleton.nodeTransform.data()));
        VK_CHECK_RESULT(skeleton.ssbo.map());


        for (auto& node : nodes)
        {
            updateNodeTransform(node);
        }
        skeleton.ssbo.copyTo(skeleton.nodeTransform.data(), sizeof(glm::mat4) * skeleton.nodeTransform.size());
    }

    void loadNode(const tinygltf::Node &inputNode, const tinygltf::Model &input, VulkanglTFModel::Node *parent, uint32_t nodeIndex, std::vector<uint32_t> &indexBuffer, std::vector<VulkanglTFModel::Vertex> &vertexBuffer) {
        VulkanglTFModel::Node *node = new VulkanglTFModel::Node{};
        node->parent                = parent;
        node->matrix                = glm::mat4(1.0f);
        node->index                 = nodeIndex;
        node->skin                  = inputNode.skin;

        // Get the local node matrix
        // It's either made up from translation, rotation, scale or a 4x4 matrix
        if (inputNode.translation.size() == 3)
        {
            node->translation = glm::make_vec3(inputNode.translation.data());
        }
        if (inputNode.rotation.size() == 4)
        {
            glm::quat q    = glm::make_quat(inputNode.rotation.data());
            node->rotation = glm::mat4(q);
        }
        if (inputNode.scale.size() == 3)
        {
            node->scale = glm::make_vec3(inputNode.scale.data());
        }
        if (inputNode.matrix.size() == 16)
        {
            node->matrix = glm::make_mat4x4(inputNode.matrix.data());
        };

        // Load node's children
        if (inputNode.children.size() > 0)
        {
            for (size_t i = 0; i < inputNode.children.size(); i++)
            {
                loadNode(input.nodes[inputNode.children[i]], input, node, inputNode.children[i], indexBuffer, vertexBuffer);
            }
        }

        // If the node contains mesh data, we load vertices and indices from the buffers
        // In glTF this is done via accessors and buffer views
        if (inputNode.mesh > -1)
        {
            const tinygltf::Mesh mesh = input.meshes[inputNode.mesh];
            // Iterate through all primitives of this node's mesh
            for (size_t i = 0; i < mesh.primitives.size(); i++)
            {
                const tinygltf::Primitive &glTFPrimitive = mesh.primitives[i];
                uint32_t                   firstIndex    = static_cast<uint32_t>(indexBuffer.size());
                uint32_t                   vertexStart   = static_cast<uint32_t>(vertexBuffer.size());
                uint32_t                   indexCount    = 0;
                bool                       hasSkin       = false;
                // Vertices
                {
                    const float *   positionBuffer     = nullptr;
                    const float *   normalsBuffer      = nullptr;
                    const float *   texCoordsBuffer    = nullptr;
                    const float *   tangentBuffer      = nullptr;
                    size_t          vertexCount        = 0;

                    // Get buffer data for vertex normals
                    if (glTFPrimitive.attributes.find("POSITION") != glTFPrimitive.attributes.end())
                    {
                        const tinygltf::Accessor &  accessor = input.accessors[glTFPrimitive.attributes.find("POSITION")->second];
                        const tinygltf::BufferView &view     = input.bufferViews[accessor.bufferView];
                        positionBuffer                       = reinterpret_cast<const float *>(&(input.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]));
                        vertexCount                          = accessor.count;
                    }
                    // Get buffer data for vertex normals
                    if (glTFPrimitive.attributes.find("NORMAL") != glTFPrimitive.attributes.end())
                    {
                        const tinygltf::Accessor &  accessor = input.accessors[glTFPrimitive.attributes.find("NORMAL")->second];
                        const tinygltf::BufferView &view     = input.bufferViews[accessor.bufferView];
                        normalsBuffer                        = reinterpret_cast<const float *>(&(input.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]));
                    }
                    // Get buffer data for vertex texture coordinates
                    // glTF supports multiple sets, we only load the first one
                    if (glTFPrimitive.attributes.find("TEXCOORD_0") != glTFPrimitive.attributes.end())
                    {
                        const tinygltf::Accessor &  accessor = input.accessors[glTFPrimitive.attributes.find("TEXCOORD_0")->second];
                        const tinygltf::BufferView &view     = input.bufferViews[accessor.bufferView];
                        texCoordsBuffer                      = reinterpret_cast<const float *>(&(input.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]));
                    }

                    // Get buffer data for vertex tangents
                    if (glTFPrimitive.attributes.find("TANGENT") != glTFPrimitive.attributes.end())
                    {
                        const tinygltf::Accessor &  accessor = input.accessors[glTFPrimitive.attributes.find("TANGENT")->second];
                        const tinygltf::BufferView &view     = input.bufferViews[accessor.bufferView];
                        tangentBuffer                        = reinterpret_cast<const float*>(&(input.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]));
                    }

                    // Append data to model's vertex buffer
                    for (size_t v = 0; v < vertexCount; v++)
                    {
                        Vertex vert{};
                        vert.pos = glm::vec4(glm::make_vec3(&positionBuffer[v * 3]), 1.0f);
                        vert.normal = glm::normalize(glm::vec3(normalsBuffer ? glm::make_vec3(&normalsBuffer[v * 3]) : glm::vec3(0.0f)));
                        vert.uv = texCoordsBuffer ? glm::make_vec2(&texCoordsBuffer[v * 2]) : glm::vec3(0.0f);
                        vert.color = glm::vec4(1.0f);
                        vert.tangent = tangentBuffer ? glm::make_vec4(&tangentBuffer[v * 4]) : glm::vec4(0.0f);
                        vertexBuffer.push_back(vert);
                    }
                }
                // Indices
                {
                    const tinygltf::Accessor &  accessor   = input.accessors[glTFPrimitive.indices];
                    const tinygltf::BufferView &bufferView = input.bufferViews[accessor.bufferView];
                    const tinygltf::Buffer &    buffer     = input.buffers[bufferView.buffer];

                    indexCount += static_cast<uint32_t>(accessor.count);

                    // glTF supports different component types of indices
                    switch (accessor.componentType)
                    {
                        case TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT: {
                            const uint32_t* buf = reinterpret_cast<const uint32_t*>(&buffer.data[accessor.byteOffset + bufferView.byteOffset]);
                            for (size_t index = 0; index < accessor.count; index++)
                            {
                                indexBuffer.push_back(buf[index] + vertexStart);
                            }
                            break;
                        }
                        case TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT: {
                            const uint16_t* buf = reinterpret_cast<const uint16_t*>(&buffer.data[accessor.byteOffset + bufferView.byteOffset]);
                            for (size_t index = 0; index < accessor.count; index++)
                            {
                                indexBuffer.push_back(buf[index] + vertexStart);
                            }
                            break;
                        }
                        case TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE: {
                            const uint8_t* buf = reinterpret_cast<const uint8_t*>(&buffer.data[accessor.byteOffset + bufferView.byteOffset]);
                            for (size_t index = 0; index < accessor.count; index++)
                            {
                                indexBuffer.push_back(buf[index] + vertexStart);
                            }
                            break;
                        }
                        default:
                            std::cerr << "Index component type " << accessor.componentType << " not supported!" << std::endl;
                            return;
                    }
                }
                Primitive primitive{};
                primitive.firstIndex    = firstIndex;
                primitive.indexCount    = indexCount;
                primitive.materialIndex = glTFPrimitive.material;
                node->mesh.primitives.push_back(primitive);
            }
        }

        if (parent)
        {
            parent->children.push_back(node);
        }
        else
        {
            nodes.push_back(node);
        }
    }

    /*
	    glTF vertex skinning functions
    */

    // POI: Traverse the node hierarchy to the top-most parent to get the local matrix of the given node
    glm::mat4 getNodeMatrix(VulkanglTFModel::Node *node)
    {
        glm::mat4              nodeMatrix    = node->getLocalMatrix();
        VulkanglTFModel::Node *currentParent = node->parent;
        while (currentParent)
        {
            nodeMatrix    = currentParent->getLocalMatrix() * nodeMatrix;
            currentParent = currentParent->parent;
        }
        return nodeMatrix;
    }

    // POI: Update the joint matrices from the current animation frame and pass them to the GPU
    void updateJoints(VulkanglTFModel::Node *node)
    {
        if (node->skin > -1)
        {
            // Update the joint matrices
            glm::mat4              inverseTransform = glm::inverse(getNodeMatrix(node));
            Skin                   skin             = skins[node->skin];
            size_t                 numJoints        = (uint32_t) skin.joints.size();
            std::vector<glm::mat4> jointMatrices(numJoints);
            for (size_t i = 0; i < numJoints; i++)
            {
                jointMatrices[i] = getNodeMatrix(skin.joints[i]) * skin.inverseBindMatrices[i];
                jointMatrices[i] = inverseTransform * jointMatrices[i];
            }
            // Update ssbo
            skin.ssbo.copyTo(jointMatrices.data(), jointMatrices.size() * sizeof(glm::mat4));
        }

        for (auto &child : node->children)
        {
            updateJoints(child);
        }
    }

    // update node transformations
    void updateNodeTransform(Node* node) {
        if (node == VK_NULL_HANDLE) {
            return;
        }

        glm::mat4 nodeMatrix = node->getLocalMatrix();
        VulkanglTFModel::Node* currentParent = node->parent;
        while (currentParent) {
            nodeMatrix = currentParent->getLocalMatrix() * nodeMatrix;
            currentParent = currentParent->parent;
        }

        skeleton.nodeTransform[node->index] = nodeMatrix;

        for (auto&& child : node->children) {
            updateNodeTransform(child);
        }
    }

    // POI: Update the current animation
    void updateAnimation(float deltaTime)
    {
        if (activeAnimation > static_cast<uint32_t>(animations.size()) - 1)
        {
            std::cout << "No animation with index " << activeAnimation << std::endl;
            return;
        }
        Animation &animation = animations[activeAnimation];
        animation.currentTime += deltaTime;
        if (animation.currentTime > animation.end)
        {
            animation.currentTime -= animation.end;
        }

        for (auto &channel : animation.channels)
        {
            AnimationSampler &sampler = animation.samplers[channel.samplerIndex];
            for (size_t i = 0; i < sampler.inputs.size() - 1; i++)
            {
                if (sampler.interpolation != "LINEAR")
                {
                    std::cout << "This sample only supports linear interpolations\n";
                    continue;
                }

                // Get the input keyframe values for the current time stamp
                if ((animation.currentTime >= sampler.inputs[i]) && (animation.currentTime <= sampler.inputs[i + 1]))
                {
                    float a = (animation.currentTime - sampler.inputs[i]) / (sampler.inputs[i + 1] - sampler.inputs[i]);
                    if (channel.path == "translation")
                    {
                        channel.node->translation = glm::mix(sampler.outputsVec4[i], sampler.outputsVec4[i + 1], a);
                    }
                    if (channel.path == "rotation")
                    {
                        glm::quat q1;
                        q1.x = sampler.outputsVec4[i].x;
                        q1.y = sampler.outputsVec4[i].y;
                        q1.z = sampler.outputsVec4[i].z;
                        q1.w = sampler.outputsVec4[i].w;

                        glm::quat q2;
                        q2.x = sampler.outputsVec4[i + 1].x;
                        q2.y = sampler.outputsVec4[i + 1].y;
                        q2.z = sampler.outputsVec4[i + 1].z;
                        q2.w = sampler.outputsVec4[i + 1].w;

                        channel.node->rotation = glm::normalize(glm::slerp(q1, q2, a));
                    }
                    if (channel.path == "scale")
                    {
                        channel.node->scale = glm::mix(sampler.outputsVec4[i], sampler.outputsVec4[i + 1], a);
                    }
                }
            }
        }

        for (auto& node: nodes) {
            updateNodeTransform(node);
        }
        skeleton.ssbo.copyTo(skeleton.nodeTransform.data(), skeleton.nodeTransform.size() * sizeof(glm::mat4));
    }

	/*
		glTF rendering functions
	*/

    // Draw a single node including child nodes (if present)
    void drawNode(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, VulkanglTFModel::Node *node) {
        if (!node->mesh.primitives.empty()) {
            // Pass the node's matrix via push constants
            // Traverse the node hierarchy to the top-most parent to get the final matrix of the current node
            glm::mat4 nodeMatrix = getNodeMatrix(node);

            // Pass the final matrix to the vertex shader using push constants
            vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), &nodeMatrix);

            // Bind SSBO with skin data for this node to set 1
            // vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 1, 1, &skins[node.skin].descriptorSet, 0, nullptr);

            for (VulkanglTFModel::Primitive &primitive : node->mesh.primitives) {
                if (primitive.indexCount > 0) {
                    // Get the texture index for this primitive
                    // VulkanglTFModel::Texture texture = textures[materials[primitive.materialIndex].baseColorTextureIndex];
                    // Bind the descriptor for the current primitive's texture to set 1

                    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 1, 1, &materials[primitive.materialIndex].descriptorSet, 0, nullptr);
                    vkCmdDrawIndexed(commandBuffer, primitive.indexCount, 1, primitive.firstIndex, 0, 0);
                }
            }
        }
        for (auto &child : node->children) {
            drawNode(commandBuffer, pipelineLayout, child);
        }
    }

    // Draw the glTF scene starting at the top-level-nodes
    void draw(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout) {
        // All vertices and indices are stored in single buffers, so we only need to bind once
        VkDeviceSize offsets[1] = {0};
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertices.buffer, offsets);
        vkCmdBindIndexBuffer(commandBuffer, indices.buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 2, 1, &skeleton.descriptorSet, 0, nullptr);

        // Render all nodes at top-level
        for (auto &node : nodes) {
            drawNode(commandBuffer, pipelineLayout, node);
        }
    }

};

class VulkanExample : public VulkanExampleBase
{
public:
	bool wireframe = false;

	VulkanglTFModel glTFModel;

    vks::Texture2D defaultOccusionMap;
    vks::Texture2D defaultEmissiveMap;

	struct ShaderData {
		vks::Buffer buffer;
		struct Values {
			glm::mat4 projection;
			glm::mat4 model;
			glm::vec4 lightPos = glm::vec4(5.0f, 5.0f, 5.0f, 1.0f);
            glm::vec4 viewPos;
		} values;
	} shaderData;

	struct Pipelines {
		VkPipeline solid;
		VkPipeline wireframe = VK_NULL_HANDLE;
	} pipelines;

	VkPipelineLayout pipelineLayout;
	VkDescriptorSet descriptorSet;

	struct DescriptorSetLayouts {
		VkDescriptorSetLayout matrices;
		VkDescriptorSetLayout textures;
        VkDescriptorSetLayout transform;
	} descriptorSetLayouts;

	VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
	{
		title = "homework1";
		camera.type = Camera::CameraType::lookat;
		camera.flipY = true;
		camera.setPosition(glm::vec3(0.0f, -0.1f, -1.0f));
		camera.setRotation(glm::vec3(0.0f, 45.0f, 0.0f));
		camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 256.0f);
	}

	~VulkanExample()
	{
		// Clean up used Vulkan resources
		// Note : Inherited destructor cleans up resources stored in base class
		vkDestroyPipeline(device, pipelines.solid, nullptr);
		if (pipelines.wireframe != VK_NULL_HANDLE) {
			vkDestroyPipeline(device, pipelines.wireframe, nullptr);
		}

		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorSetLayouts.matrices, nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorSetLayouts.textures, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayouts.transform, nullptr);

		shaderData.buffer.destroy();
	}

	virtual void getEnabledFeatures()
	{
		// Fill mode non solid is required for wireframe display
		if (deviceFeatures.fillModeNonSolid) {
			enabledFeatures.fillModeNonSolid = VK_TRUE;
		};
	}

	void buildCommandBuffers()
	{
		VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		VkClearValue clearValues[2];
		clearValues[0].color = defaultClearColor;
		clearValues[0].color = { { 0.25f, 0.25f, 0.25f, 1.0f } };;
		clearValues[1].depthStencil = { 1.0f, 0 };

		VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
		renderPassBeginInfo.renderPass = renderPass;
		renderPassBeginInfo.renderArea.offset.x = 0;
		renderPassBeginInfo.renderArea.offset.y = 0;
		renderPassBeginInfo.renderArea.extent.width = width;
		renderPassBeginInfo.renderArea.extent.height = height;
		renderPassBeginInfo.clearValueCount = 2;
		renderPassBeginInfo.pClearValues = clearValues;

		const VkViewport viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
		const VkRect2D scissor = vks::initializers::rect2D(width, height, 0, 0);

		for (int32_t i = 0; i < drawCmdBuffers.size(); ++i)
		{
			renderPassBeginInfo.framebuffer = frameBuffers[i];
			VK_CHECK_RESULT(vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufInfo));
			vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
			vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);
			vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);
			// Bind scene matrices descriptor to set 0
			vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
			vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, wireframe ? pipelines.wireframe : pipelines.solid);
			glTFModel.draw(drawCmdBuffers[i], pipelineLayout);
			drawUI(drawCmdBuffers[i]);
			vkCmdEndRenderPass(drawCmdBuffers[i]);
			VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
		}
	}

	void loadglTFFile(std::string filename)
	{
		tinygltf::Model glTFInput;
		tinygltf::TinyGLTF gltfContext;
		std::string error, warning;

		this->device = device;

#if defined(__ANDROID__)
		// On Android all assets are packed with the apk in a compressed form, so we need to open them using the asset manager
		// We let tinygltf handle this, by passing the asset manager of our app
		tinygltf::asset_manager = androidApp->activity->assetManager;
#endif
		bool fileLoaded = gltfContext.LoadASCIIFromFile(&glTFInput, &error, &warning, filename);

		// Pass some Vulkan resources required for setup and rendering to the glTF model loading class
		glTFModel.vulkanDevice = vulkanDevice;
		glTFModel.copyQueue = queue;

		std::vector<uint32_t> indexBuffer;
		std::vector<VulkanglTFModel::Vertex> vertexBuffer;

		if (fileLoaded) {
			glTFModel.loadImages(glTFInput);
			glTFModel.loadMaterials(glTFInput);
			glTFModel.loadTextures(glTFInput);
			const tinygltf::Scene& scene = glTFInput.scenes[0];
			for (size_t i = 0; i < scene.nodes.size(); i++) {
				const tinygltf::Node node = glTFInput.nodes[scene.nodes[i]];
				glTFModel.loadNode(node, glTFInput, nullptr, scene.nodes[i], indexBuffer, vertexBuffer);
			}

            glTFModel.loadAnimations(glTFInput);
            glTFModel.prepareAnimations(glTFInput);
		}
		else {
			vks::tools::exitFatal("Could not open the glTF file.\n\nThe file is part of the additional asset pack.\n\nRun \"download_assets.py\" in the repository root to download the latest version.", -1);
			return;
		}

		// Create and upload vertex and index buffer
		// We will be using one single vertex buffer and one single index buffer for the whole glTF scene
		// Primitives (of the glTF model) will then index into these using index offsets

		size_t vertexBufferSize = vertexBuffer.size() * sizeof(VulkanglTFModel::Vertex);
		size_t indexBufferSize = indexBuffer.size() * sizeof(uint32_t);
		glTFModel.indices.count = static_cast<uint32_t>(indexBuffer.size());

		struct StagingBuffer {
			VkBuffer buffer;
			VkDeviceMemory memory;
		} vertexStaging, indexStaging;

		// Create host visible staging buffers (source)
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			vertexBufferSize,
			&vertexStaging.buffer,
			&vertexStaging.memory,
			vertexBuffer.data()));
		// Index data
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			indexBufferSize,
			&indexStaging.buffer,
			&indexStaging.memory,
			indexBuffer.data()));

		// Create device local buffers (target)
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			vertexBufferSize,
			&glTFModel.vertices.buffer,
			&glTFModel.vertices.memory));
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			indexBufferSize,
			&glTFModel.indices.buffer,
			&glTFModel.indices.memory));

		// Copy data from staging buffers (host) do device local buffer (gpu)
		VkCommandBuffer copyCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		VkBufferCopy copyRegion = {};

		copyRegion.size = vertexBufferSize;
		vkCmdCopyBuffer(
			copyCmd,
			vertexStaging.buffer,
			glTFModel.vertices.buffer,
			1,
			&copyRegion);

		copyRegion.size = indexBufferSize;
		vkCmdCopyBuffer(
			copyCmd,
			indexStaging.buffer,
			glTFModel.indices.buffer,
			1,
			&copyRegion);

		vulkanDevice->flushCommandBuffer(copyCmd, queue, true);

		// Free staging resources
		vkDestroyBuffer(device, vertexStaging.buffer, nullptr);
		vkFreeMemory(device, vertexStaging.memory, nullptr);
		vkDestroyBuffer(device, indexStaging.buffer, nullptr);
		vkFreeMemory(device, indexStaging.memory, nullptr);
	}

	void loadAssets()
	{
		loadglTFFile(getAssetPath() + "buster_drone/busterDrone.gltf");
	}

    void loadDefaultTextureMap() {
        uint32_t width = 2048, height = 2048;
        std::vector<unsigned char> buffer(width * height * 4, 1);
        auto bufferSize = static_cast<VkDeviceSize>(buffer.size());
        defaultOccusionMap.fromBuffer(buffer.data(), bufferSize, VK_FORMAT_R8G8B8A8_UNORM, width, height, vulkanDevice, queue);

        buffer.resize(width * height * 4, 0);
        defaultEmissiveMap.fromBuffer(buffer.data(), bufferSize, VK_FORMAT_R8G8B8A8_UNORM, width, height, vulkanDevice, queue);
    }

	void setupDescriptors()
	{
		/*
			This sample uses separate descriptor sets (and layouts) for the matrices and materials (textures)
		*/

		std::vector<VkDescriptorPoolSize> poolSizes = {
            // One uniform buffer block for each model's texture
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, static_cast<uint32_t>(2 + glTFModel.materials.size())),
			// Combined image sampler per model image/texture ()
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, static_cast<uint32_t>(5 * glTFModel.materials.size())),
            // ssbo for animation data
            vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1),
		};
		// One set for matrices and one per model image/texture
		const uint32_t maxSetCount = static_cast<uint32_t>(glTFModel.images.size()) + 2;
		VkDescriptorPoolCreateInfo descriptorPoolInfo = vks::initializers::descriptorPoolCreateInfo(poolSizes, maxSetCount);
		VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));

        // Descriptor set layouts
        VkDescriptorSetLayoutBinding setLayoutBinding{};
        VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(&setLayoutBinding, 1);

		// Descriptor set layout for passing matrices
		setLayoutBinding = vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0);
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCI, nullptr, &descriptorSetLayouts.matrices));

        // Descriptor set layout for passing node transformations
        setLayoutBinding = vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0);
        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCI, nullptr, &descriptorSetLayouts.transform));

		// Descriptor set layout for passing material textures
        std::vector<VkDescriptorSetLayoutBinding> SetLayoutBindingForTexture = {
                vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0),
                vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1),
                vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 2),
                vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 3),
                vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 4),
        };
        descriptorSetLayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(SetLayoutBindingForTexture);
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCI, nullptr, &descriptorSetLayouts.textures))

        // Pipeline layout using both descriptor sets (set 0 = matrices, set 1 = textures, set 2 = model transform)
		std::array<VkDescriptorSetLayout, 3> setLayouts = {
                descriptorSetLayouts.matrices,
                descriptorSetLayouts.textures,
                descriptorSetLayouts.transform,
        };
		VkPipelineLayoutCreateInfo pipelineLayoutCI = vks::initializers::pipelineLayoutCreateInfo(setLayouts.data(), static_cast<uint32_t>(setLayouts.size()));

        // We will use push constants to push the local matrices of a primitive to the vertex shader
		VkPushConstantRange pushConstantRange = vks::initializers::pushConstantRange(VK_SHADER_STAGE_VERTEX_BIT, sizeof(glm::mat4), 0);
		// Push constant ranges are part of the pipeline layout
		pipelineLayoutCI.pushConstantRangeCount = 1;
		pipelineLayoutCI.pPushConstantRanges = &pushConstantRange;
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCI, nullptr, &pipelineLayout));

		// Descriptor set for scene matrices
		VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayouts.matrices, 1);
		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet));
		VkWriteDescriptorSet writeDescriptorSet = vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &shaderData.buffer.descriptor);
		vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, nullptr);

        // Descriptor sets for glTF model materials
        allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayouts.textures, 1);
        for (auto& material : glTFModel.materials) {
            VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &material.descriptorSet))

            std::array<VkDescriptorImageInfo*, 5> descriptorImageInfos{};
            if (material.baseColorTextureIndex != -1) {
                descriptorImageInfos[0] = &glTFModel.images[material.baseColorTextureIndex].texture.descriptor;
            }
            if (material.metallicRoughnessTextureIndex != -1) {
                descriptorImageInfos[1] = &glTFModel.images[material.metallicRoughnessTextureIndex].texture.descriptor;
            }
            if (material.normalTextureIndex != -1) {
                descriptorImageInfos[2] = &glTFModel.images[material.normalTextureIndex].texture.descriptor;
            }
            if (material.emissiveTextureIndex != -1) {
                descriptorImageInfos[3] = &glTFModel.images[material.emissiveTextureIndex].texture.descriptor;
            }
            else {
                descriptorImageInfos[3] = &defaultEmissiveMap.descriptor;
            }
            if (material.occlusionTextureIndex != -1) {
                descriptorImageInfos[4] = &glTFModel.images[material.occlusionTextureIndex].texture.descriptor;
            }
            else {
                descriptorImageInfos[4] = &defaultOccusionMap.descriptor;
            }
            std::vector<VkWriteDescriptorSet> writeDescriptorSets;
            for (int i = 0; i < descriptorImageInfos.size(); ++i) {
                if (descriptorImageInfos[i] != nullptr) {
                    VkWriteDescriptorSet w =  vks::initializers::writeDescriptorSet(material.descriptorSet, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, i, descriptorImageInfos[i],  1);
                    writeDescriptorSets.emplace_back(w);
                }
            }
            vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
        }

        // Descriptor set for node matrix
        allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayouts.transform, 1);
        VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &glTFModel.skeleton.descriptorSet));
        writeDescriptorSet = vks::initializers::writeDescriptorSet(glTFModel.skeleton.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0, &glTFModel.skeleton.ssbo.descriptor);
        vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, nullptr);
	}

	void preparePipelines()
	{
		VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCI = vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
		VkPipelineRasterizationStateCreateInfo rasterizationStateCI = vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0);
		VkPipelineColorBlendAttachmentState blendAttachmentStateCI = vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
		VkPipelineColorBlendStateCreateInfo colorBlendStateCI = vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentStateCI);
		VkPipelineDepthStencilStateCreateInfo depthStencilStateCI = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL);
		VkPipelineViewportStateCreateInfo viewportStateCI = vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);
		VkPipelineMultisampleStateCreateInfo multisampleStateCI = vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT, 0);
		const std::vector<VkDynamicState> dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
		VkPipelineDynamicStateCreateInfo dynamicStateCI = vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables.data(), static_cast<uint32_t>(dynamicStateEnables.size()), 0);
		// Vertex input bindings and attributes
		const std::vector<VkVertexInputBindingDescription> vertexInputBindings = {
			vks::initializers::vertexInputBindingDescription(0, sizeof(VulkanglTFModel::Vertex), VK_VERTEX_INPUT_RATE_VERTEX),
		};
		const std::vector<VkVertexInputAttributeDescription> vertexInputAttributes = {
                {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VulkanglTFModel::Vertex, pos)},
                {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VulkanglTFModel::Vertex, normal)},
                {2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VulkanglTFModel::Vertex, uv)},
                {3, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(VulkanglTFModel::Vertex, color)},
                {4, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(VulkanglTFModel::Vertex, tangent)},
		};

		VkPipelineVertexInputStateCreateInfo vertexInputStateCI = vks::initializers::pipelineVertexInputStateCreateInfo();
		vertexInputStateCI.vertexBindingDescriptionCount = static_cast<uint32_t>(vertexInputBindings.size());
		vertexInputStateCI.pVertexBindingDescriptions = vertexInputBindings.data();
		vertexInputStateCI.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInputAttributes.size());
		vertexInputStateCI.pVertexAttributeDescriptions = vertexInputAttributes.data();

		const std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages = {
			loadShader(getHomeworkShadersPath() + "homework1/mesh.vert.spv", VK_SHADER_STAGE_VERTEX_BIT),
			loadShader(getHomeworkShadersPath() + "homework1/mesh.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT)
		};

		VkGraphicsPipelineCreateInfo pipelineCI = vks::initializers::pipelineCreateInfo(pipelineLayout, renderPass, 0);
		pipelineCI.pVertexInputState = &vertexInputStateCI;
		pipelineCI.pInputAssemblyState = &inputAssemblyStateCI;
		pipelineCI.pRasterizationState = &rasterizationStateCI;
		pipelineCI.pColorBlendState = &colorBlendStateCI;
		pipelineCI.pMultisampleState = &multisampleStateCI;
		pipelineCI.pViewportState = &viewportStateCI;
		pipelineCI.pDepthStencilState = &depthStencilStateCI;
		pipelineCI.pDynamicState = &dynamicStateCI;
		pipelineCI.stageCount = static_cast<uint32_t>(shaderStages.size());
		pipelineCI.pStages = shaderStages.data();

		// Solid rendering pipeline
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipelines.solid));

		// Wire frame rendering pipeline
		if (deviceFeatures.fillModeNonSolid) {
			rasterizationStateCI.polygonMode = VK_POLYGON_MODE_LINE;
			rasterizationStateCI.lineWidth = 1.0f;
			VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipelines.wireframe));
		}
	}

	// Prepare and initialize uniform buffer containing shader uniforms
	void prepareUniformBuffers()
	{
		// Vertex shader uniform buffer block
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&shaderData.buffer,
			sizeof(shaderData.values)));

		// Map persistent
		VK_CHECK_RESULT(shaderData.buffer.map());

		updateUniformBuffers();
	}

	void updateUniformBuffers()
	{
		shaderData.values.projection = camera.matrices.perspective;
		shaderData.values.model = camera.matrices.view;
        shaderData.values.viewPos = camera.viewPos;
		memcpy(shaderData.buffer.mapped, &shaderData.values, sizeof(shaderData.values));
	}

	void prepare()
	{
		VulkanExampleBase::prepare();
        loadDefaultTextureMap();
        loadAssets();

		prepareUniformBuffers();
		setupDescriptors();
		preparePipelines();
		buildCommandBuffers();
		prepared = true;
	}

	virtual void render()
	{
		renderFrame();
		if (camera.updated) {
			updateUniformBuffers();
		}

        // POI: Advance animation
        if (!paused) {
            glTFModel.updateAnimation(frameTimer);
            buildCommandBuffers();
        }
	}

	virtual void viewChanged()
	{
		updateUniformBuffers();
	}

	virtual void OnUpdateUIOverlay(vks::UIOverlay *overlay)
	{
		if (overlay->header("Settings")) {
			if (overlay->checkBox("Wireframe", &wireframe)) {
				buildCommandBuffers();
			}
		}
	}
};

VULKAN_EXAMPLE_MAIN()
