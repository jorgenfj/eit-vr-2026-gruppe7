#include "fbx_exporter.h"

#include <assimp/Importer.hpp>
#include <assimp/Exporter.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cstring>
#include <iostream>
#include <vector>

// ── Helper: build a mesh of thin cylinders from GL_LINES pairs ──────────────
// Each line segment (A→B) becomes a quad ribbon (2 triangles) with a small
// width so it's visible in Blender.  We orient the ribbon to face "up" or
// use the cross product with an arbitrary axis.

static aiMesh* buildLineMesh(const std::vector<Vertex>& lineVerts,
                             const char* name,
                             unsigned materialIndex,
                             float ribbonWidth = 0.003f)
{
    size_t numSegs = lineVerts.size() / 2;
    if (numSegs == 0) return nullptr;

    aiMesh* mesh = new aiMesh();
    mesh->mName = aiString(name);
    mesh->mMaterialIndex = materialIndex;
    mesh->mPrimitiveTypes = aiPrimitiveType_TRIANGLE;

    // Each segment → 4 vertices, 2 triangles (6 indices)
    mesh->mNumVertices = (unsigned)(numSegs * 4);
    mesh->mVertices = new aiVector3D[mesh->mNumVertices];
    mesh->mNormals  = new aiVector3D[mesh->mNumVertices];

    mesh->mNumFaces = (unsigned)(numSegs * 2);
    mesh->mFaces = new aiFace[mesh->mNumFaces];

    for (size_t s = 0; s < numSegs; ++s) {
        const glm::vec3& a = lineVerts[s * 2 + 0].pos;
        const glm::vec3& b = lineVerts[s * 2 + 1].pos;

        glm::vec3 dir = b - a;
        float len = glm::length(dir);
        if (len < 1e-8f) {
            dir = glm::vec3(0.f, 0.f, 1.f);
            len = 1.f;
        }
        dir /= len;

        // Find a perpendicular direction for ribbon width
        glm::vec3 up = (std::fabs(dir.y) < 0.99f)
                            ? glm::vec3(0.f, 1.f, 0.f)
                            : glm::vec3(1.f, 0.f, 0.f);
        glm::vec3 side = glm::normalize(glm::cross(dir, up)) * ribbonWidth;

        // 4 vertices: A-side, A+side, B+side, B-side
        unsigned base = (unsigned)(s * 4);
        glm::vec3 v0 = a - side;
        glm::vec3 v1 = a + side;
        glm::vec3 v2 = b + side;
        glm::vec3 v3 = b - side;

        mesh->mVertices[base + 0] = aiVector3D(v0.x, v0.y, v0.z);
        mesh->mVertices[base + 1] = aiVector3D(v1.x, v1.y, v1.z);
        mesh->mVertices[base + 2] = aiVector3D(v2.x, v2.y, v2.z);
        mesh->mVertices[base + 3] = aiVector3D(v3.x, v3.y, v3.z);

        // Normal (face normal of the ribbon)
        glm::vec3 n = glm::normalize(glm::cross(v1 - v0, v3 - v0));
        for (int i = 0; i < 4; ++i)
            mesh->mNormals[base + i] = aiVector3D(n.x, n.y, n.z);

        // Two triangles: (0,1,2) and (0,2,3)
        unsigned fbase = (unsigned)(s * 2);
        mesh->mFaces[fbase + 0].mNumIndices = 3;
        mesh->mFaces[fbase + 0].mIndices = new unsigned[3];
        mesh->mFaces[fbase + 0].mIndices[0] = base + 0;
        mesh->mFaces[fbase + 0].mIndices[1] = base + 1;
        mesh->mFaces[fbase + 0].mIndices[2] = base + 2;

        mesh->mFaces[fbase + 1].mNumIndices = 3;
        mesh->mFaces[fbase + 1].mIndices = new unsigned[3];
        mesh->mFaces[fbase + 1].mIndices[0] = base + 0;
        mesh->mFaces[fbase + 1].mIndices[1] = base + 2;
        mesh->mFaces[fbase + 1].mIndices[2] = base + 3;
    }

    return mesh;
}

// ── Helper: build a triangle mesh directly from vertex triples ───────────────
static aiMesh* buildTriMesh(const std::vector<Vertex>& triVerts,
                            const char* name,
                            unsigned materialIndex)
{
    size_t numTris = triVerts.size() / 3;
    if (numTris == 0) return nullptr;

    aiMesh* mesh = new aiMesh();
    mesh->mName = aiString(name);
    mesh->mMaterialIndex = materialIndex;
    mesh->mPrimitiveTypes = aiPrimitiveType_TRIANGLE;

    mesh->mNumVertices = (unsigned)triVerts.size();
    mesh->mVertices = new aiVector3D[mesh->mNumVertices];
    mesh->mNormals  = new aiVector3D[mesh->mNumVertices];

    mesh->mNumFaces = (unsigned)numTris;
    mesh->mFaces = new aiFace[mesh->mNumFaces];

    for (size_t t = 0; t < numTris; ++t) {
        unsigned base = (unsigned)(t * 3);
        const glm::vec3& a = triVerts[base + 0].pos;
        const glm::vec3& b = triVerts[base + 1].pos;
        const glm::vec3& c = triVerts[base + 2].pos;

        mesh->mVertices[base + 0] = aiVector3D(a.x, a.y, a.z);
        mesh->mVertices[base + 1] = aiVector3D(b.x, b.y, b.z);
        mesh->mVertices[base + 2] = aiVector3D(c.x, c.y, c.z);

        glm::vec3 n = glm::normalize(glm::cross(b - a, c - a));
        for (int i = 0; i < 3; ++i)
            mesh->mNormals[base + i] = aiVector3D(n.x, n.y, n.z);

        mesh->mFaces[t].mNumIndices = 3;
        mesh->mFaces[t].mIndices = new unsigned[3];
        mesh->mFaces[t].mIndices[0] = base + 0;
        mesh->mFaces[t].mIndices[1] = base + 1;
        mesh->mFaces[t].mIndices[2] = base + 2;
    }

    return mesh;
}

// ── Helper: create a simple material with color + opacity ───────────────────

static aiMaterial* buildColorMaterial(const char* name,
                                     const glm::vec3& color,
                                     float alpha)
{
    aiMaterial* mat = new aiMaterial();

    aiString matName(name);
    mat->AddProperty(&matName, AI_MATKEY_NAME);

    aiColor3D diffuse(color.r, color.g, color.b);
    mat->AddProperty(&diffuse, 1, AI_MATKEY_COLOR_DIFFUSE);

    aiColor3D emissive(color.r * 0.3f, color.g * 0.3f, color.b * 0.3f);
    mat->AddProperty(&emissive, 1, AI_MATKEY_COLOR_EMISSIVE);

    float opacity = alpha;
    mat->AddProperty(&opacity, 1, AI_MATKEY_OPACITY);

    // Blender reads the shading model for import
    int shadingModel = aiShadingMode_Gouraud;
    mat->AddProperty(&shadingModel, 1, AI_MATKEY_SHADING_MODEL);

    return mat;
}

// ── Main export function ────────────────────────────────────────────────────

bool exportSceneFbx(const std::string& originalFbxPath,
                    const std::string& outputPath,
                    const std::vector<Vertex>& rayVerts,
                    const glm::vec3& rayColor,
                    float rayAlpha,
                    const std::vector<Vertex>& outlineVerts,
                    const glm::vec3& outlineColor,
                    float outlineAlpha,
                    float meshScale)
{
    // ── Scale ray/outline vertices back to original FBX coordinate system ─
    // The app scales the mesh by meshScale (e.g. 0.01 for cm→m) on load.
    // Ray/outline positions live in that scaled space, but the exported FBX
    // keeps the original unscaled mesh, so we undo the scale here.
    const float invScale = (meshScale > 1e-8f) ? (1.0f / meshScale) : 1.0f;

    auto rescaleVerts = [&](const std::vector<Vertex>& src) {
        std::vector<Vertex> dst = src;
        for (auto& v : dst)
            v.pos *= invScale;
        return dst;
    };

    std::vector<Vertex> scaledRayVerts     = rescaleVerts(rayVerts);
    std::vector<Vertex> scaledOutlineVerts = rescaleVerts(outlineVerts);

    // ── Step 1: Import the original FBX ──────────────────────────────────
    Assimp::Importer importer;
    const aiScene* srcScene = importer.ReadFile(
        originalFbxPath,
        aiProcess_Triangulate | aiProcess_GenNormals);

    if (!srcScene || !srcScene->mRootNode) {
        std::cerr << "Export: failed to re-import original FBX: "
                  << importer.GetErrorString() << "\n";
        return false;
    }

    // ── Step 2: Build a brand new scene ──────────────────────────────────
    // We deep-copy the source scene and add our new meshes + materials.
    // Assimp::Exporter needs an aiScene* it can work with.
    //
    // Since Assimp's scene copy is complex, the simplest approach is to
    // create a new scene that references the original data plus our additions.
    // Unfortunately Assimp doesn't provide a scene-copy API, so we build
    // a fresh scene with just the original meshes + our additions.

    aiScene* scene = new aiScene();

    // Count how many meshes and materials we'll have
    unsigned numOrigMeshes = srcScene->mNumMeshes;
    unsigned numOrigMats   = srcScene->mNumMaterials;

    // How many new objects? (rays mesh + outline mesh, each if non-empty)
    bool hasRays    = (rayVerts.size() >= 2);
    bool hasOutline = (outlineVerts.size() >= 2);
    unsigned numNewMeshes = (hasRays ? 1 : 0) + (hasOutline ? 1 : 0);
    unsigned numNewMats   = numNewMeshes;

    scene->mNumMeshes = numOrigMeshes + numNewMeshes;
    scene->mMeshes = new aiMesh*[scene->mNumMeshes];

    scene->mNumMaterials = numOrigMats + numNewMats;
    scene->mMaterials = new aiMaterial*[scene->mNumMaterials];

    // Copy original meshes (shallow — just pointer copy; Assimp Exporter
    // reads but doesn't free them, the Importer owns the originals)
    for (unsigned i = 0; i < numOrigMeshes; ++i)
        scene->mMeshes[i] = srcScene->mMeshes[i];

    // Copy original materials
    for (unsigned i = 0; i < numOrigMats; ++i)
        scene->mMaterials[i] = srcScene->mMaterials[i];

    // Add new materials + meshes
    unsigned nextMeshIdx = numOrigMeshes;
    unsigned nextMatIdx  = numOrigMats;

    unsigned rayMeshIdx = 0, outlineMeshIdx = 0;

    if (hasRays) {
        scene->mMaterials[nextMatIdx] = buildColorMaterial(
            "CrackRays_Material", rayColor, rayAlpha);
        aiMesh* m = buildLineMesh(scaledRayVerts, "CrackRays", nextMatIdx,
                                  0.2f * invScale);
        scene->mMeshes[nextMeshIdx] = m;
        rayMeshIdx = nextMeshIdx;
        ++nextMeshIdx;
        ++nextMatIdx;
    }

    if (hasOutline) {
        scene->mMaterials[nextMatIdx] = buildColorMaterial(
            "CrackOutline_Material", outlineColor, outlineAlpha);
        aiMesh* m = buildTriMesh(scaledOutlineVerts, "CrackOutline", nextMatIdx);
        scene->mMeshes[nextMeshIdx] = m;
        outlineMeshIdx = nextMeshIdx;
        ++nextMeshIdx;
        ++nextMatIdx;
    }

    // ── Step 3: Build the node tree ──────────────────────────────────────
    // Copy the original root node tree, then add child nodes for our meshes.

    // Deep-copy a node tree
    struct NodeCopier {
        static aiNode* copy(const aiNode* src, aiNode* parent) {
            aiNode* dst = new aiNode();
            dst->mName = src->mName;
            dst->mTransformation = src->mTransformation;
            dst->mParent = parent;

            dst->mNumMeshes = src->mNumMeshes;
            if (dst->mNumMeshes > 0) {
                dst->mMeshes = new unsigned[dst->mNumMeshes];
                std::memcpy(dst->mMeshes, src->mMeshes,
                            dst->mNumMeshes * sizeof(unsigned));
            }

            dst->mNumChildren = src->mNumChildren;
            if (dst->mNumChildren > 0) {
                dst->mChildren = new aiNode*[dst->mNumChildren];
                for (unsigned i = 0; i < dst->mNumChildren; ++i)
                    dst->mChildren[i] = copy(src->mChildren[i], dst);
            }

            return dst;
        }
    };

    aiNode* root = NodeCopier::copy(srcScene->mRootNode, nullptr);

    // Add new child nodes for rays and outline
    unsigned numNewChildren = (hasRays ? 1 : 0) + (hasOutline ? 1 : 0);
    if (numNewChildren > 0) {
        // Expand children array
        unsigned oldNumChildren = root->mNumChildren;
        unsigned newNumChildren = oldNumChildren + numNewChildren;
        aiNode** newChildren = new aiNode*[newNumChildren];
        for (unsigned i = 0; i < oldNumChildren; ++i)
            newChildren[i] = root->mChildren[i];
        delete[] root->mChildren;
        root->mChildren = newChildren;
        root->mNumChildren = newNumChildren;

        unsigned childIdx = oldNumChildren;

        if (hasRays) {
            aiNode* node = new aiNode();
            node->mName = aiString("CrackRays");
            node->mParent = root;
            node->mTransformation = aiMatrix4x4(); // identity
            node->mNumMeshes = 1;
            node->mMeshes = new unsigned[1];
            node->mMeshes[0] = rayMeshIdx;
            root->mChildren[childIdx++] = node;
        }

        if (hasOutline) {
            aiNode* node = new aiNode();
            node->mName = aiString("CrackOutline");
            node->mParent = root;
            node->mTransformation = aiMatrix4x4(); // identity
            node->mNumMeshes = 1;
            node->mMeshes = new unsigned[1];
            node->mMeshes[0] = outlineMeshIdx;
            root->mChildren[childIdx++] = node;
        }
    }

    scene->mRootNode = root;

    // ── Step 4: Export ───────────────────────────────────────────────────
    Assimp::Exporter exporter;

    // Find the FBX format ID
    const char* formatId = "fbx";
    for (size_t i = 0; i < exporter.GetExportFormatCount(); ++i) {
        const aiExportFormatDesc* desc = exporter.GetExportFormatDescription(i);
        if (desc && std::string(desc->fileExtension) == "fbx") {
            formatId = desc->id;
            break;
        }
    }

    aiReturn result = exporter.Export(scene, formatId, outputPath);

    // Clean up: we must NOT delete the original meshes/materials (owned by importer),
    // but we DO need to clean up our new ones and the node tree.
    // Set original pointers to nullptr so the scene destructor doesn't free them.
    for (unsigned i = 0; i < numOrigMeshes; ++i)
        scene->mMeshes[i] = nullptr;
    for (unsigned i = 0; i < numOrigMats; ++i)
        scene->mMaterials[i] = nullptr;

    // The scene destructor will free our new meshes, materials, and node tree.
    delete scene;

    if (result != aiReturn_SUCCESS) {
        std::cerr << "Export failed: " << exporter.GetErrorString() << "\n";
        return false;
    }

    std::cout << "Saved FBX to: " << outputPath << "\n";
    return true;
}
