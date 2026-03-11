#include "fbx_loader.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <iostream>

// ── Helpers ─────────────────────────────────────────────────────────────────

static glm::mat4 aiToGlm(const aiMatrix4x4& m)
{
    // Assimp is row-major, GLM is column-major – transpose.
    return glm::transpose(glm::make_mat4(&m.a1));
}

// ── Camera collection ───────────────────────────────────────────────────────

static void collectCameraNodes(const aiScene* scene,
                               const aiNode* node,
                               const aiMatrix4x4& parent,
                               std::vector<CameraPose>& poses)
{
    aiMatrix4x4 global = parent * node->mTransformation;
    std::string nodeName(node->mName.C_Str());

    for (unsigned i = 0; i < scene->mNumCameras; ++i) {
        if (nodeName == scene->mCameras[i]->mName.C_Str()) {
            const aiCamera* cam = scene->mCameras[i];
            glm::vec3 look(cam->mLookAt.x, cam->mLookAt.y, cam->mLookAt.z);
            glm::vec3 up  (cam->mUp.x,     cam->mUp.y,     cam->mUp.z);
            poses.push_back({nodeName, aiToGlm(global), look, up});
            break;
        }
    }

    for (unsigned i = 0; i < node->mNumChildren; ++i)
        collectCameraNodes(scene, node->mChildren[i], global, poses);
}

// ── Mesh wireframe collection ───────────────────────────────────────────────

static void collectMeshEdges(const aiScene* scene,
                             const aiNode* node,
                             const aiMatrix4x4& parent,
                             std::vector<Vertex>& verts)
{
    aiMatrix4x4 global = parent * node->mTransformation;
    glm::mat4 M = aiToGlm(global);
    glm::vec3 meshCol(0.5f, 0.5f, 0.6f);

    for (unsigned mi = 0; mi < node->mNumMeshes; ++mi) {
        const aiMesh* mesh = scene->mMeshes[node->mMeshes[mi]];
        for (unsigned fi = 0; fi < mesh->mNumFaces; ++fi) {
            const aiFace& f = mesh->mFaces[fi];
            for (unsigned ei = 0; ei < f.mNumIndices; ++ei) {
                unsigned i0 = f.mIndices[ei];
                unsigned i1 = f.mIndices[(ei + 1) % f.mNumIndices];
                glm::vec3 p0 = glm::vec3(M * glm::vec4(
                    mesh->mVertices[i0].x, mesh->mVertices[i0].y, mesh->mVertices[i0].z, 1.f));
                glm::vec3 p1 = glm::vec3(M * glm::vec4(
                    mesh->mVertices[i1].x, mesh->mVertices[i1].y, mesh->mVertices[i1].z, 1.f));
                verts.push_back({p0, meshCol});
                verts.push_back({p1, meshCol});
            }
        }
    }

    for (unsigned i = 0; i < node->mNumChildren; ++i)
        collectMeshEdges(scene, node->mChildren[i], global, verts);
}

// ── Mesh triangle collection ─────────────────────────────────────────────────

static void collectMeshTriangles(const aiScene* scene,
                                 const aiNode* node,
                                 const aiMatrix4x4& parent,
                                 std::vector<MeshVertex>& tris)
{
    aiMatrix4x4 global = parent * node->mTransformation;
    glm::mat4 M  = aiToGlm(global);
    glm::mat3 N  = glm::transpose(glm::inverse(glm::mat3(M))); // normal matrix

    for (unsigned mi = 0; mi < node->mNumMeshes; ++mi) {
        const aiMesh* mesh = scene->mMeshes[node->mMeshes[mi]];
        bool hasUV = mesh->HasTextureCoords(0);
        bool hasN  = mesh->HasNormals();

        for (unsigned fi = 0; fi < mesh->mNumFaces; ++fi) {
            const aiFace& f = mesh->mFaces[fi];
            if (f.mNumIndices != 3) continue;

            for (unsigned vi = 0; vi < 3; ++vi) {
                unsigned idx = f.mIndices[vi];
                glm::vec3 pos = glm::vec3(M * glm::vec4(
                    mesh->mVertices[idx].x,
                    mesh->mVertices[idx].y,
                    mesh->mVertices[idx].z, 1.f));

                glm::vec3 norm(0.f, 1.f, 0.f);
                if (hasN)
                    norm = glm::normalize(N * glm::vec3(
                        mesh->mNormals[idx].x,
                        mesh->mNormals[idx].y,
                        mesh->mNormals[idx].z));

                glm::vec2 uv(0.f);
                if (hasUV)
                    uv = glm::vec2(mesh->mTextureCoords[0][idx].x,
                                   mesh->mTextureCoords[0][idx].y);

                tris.push_back({pos, norm, uv});
            }
        }
    }

    for (unsigned i = 0; i < node->mNumChildren; ++i)
        collectMeshTriangles(scene, node->mChildren[i], global, tris);
}

// ── Texture path resolution ───────────────────────────────────────────────────
// The FBX may reference a texture with an absolute Windows path.
// We extract just the filename and look for it next to the FBX file.

static void resolveTexturePath(const aiScene* scene,
                               const std::string& fbxPath,
                               SceneData& out)
{
    if (scene->mNumMaterials == 0) return;
    const aiMaterial* mat = scene->mMaterials[0];
    if (mat->GetTextureCount(aiTextureType_DIFFUSE) == 0) return;

    aiString texPath;
    mat->GetTexture(aiTextureType_DIFFUSE, 0, &texPath);
    std::string raw = texPath.C_Str();

    // Extract just the filename (handles both / and \ separators)
    size_t slash = raw.find_last_of("/\\");
    std::string filename = (slash != std::string::npos) ? raw.substr(slash + 1) : raw;

    // Build path relative to the FBX directory
    size_t dirEnd = fbxPath.find_last_of("/\\");
    std::string dir = (dirEnd != std::string::npos) ? fbxPath.substr(0, dirEnd + 1) : "./";
    out.texturePath = dir + filename;

    std::cout << "  Texture path: " << out.texturePath << "\n";
}

// ── Public API ──────────────────────────────────────────────────────────────

bool loadFbx(const std::string& path, SceneData& out, float& meshScale)
{
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(
        path,
        aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs);

    if (!scene || (scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) || !scene->mRootNode) {
        std::cerr << "ERROR loading FBX: " << importer.GetErrorString() << "\n";
        return false;
    }

    // ── Auto-detect unit scale from FBX metadata ─────────────────────────
    // FBX stores UnitScaleFactor = centimetres-per-unit.
    //   1.0  → mesh is in centimetres  → scale = 0.01 to reach metres
    //   100.0 → mesh is in metres       → scale = 1.0
    //   0.1  → mesh is in millimetres  → scale = 0.001
    // If the user passed an explicit meshScale (!= 1.0) that overrides auto.
    if (meshScale == 1.0f && scene->mMetaData) {
        double unitScaleFactor = 0.0;
        bool gotUSF = false;

        // Assimp may store the value as double, float, or int — try each
        {
            double dv = 0; float fv = 0; int iv = 0;
            if (scene->mMetaData->Get("UnitScaleFactor", dv))
                { unitScaleFactor = dv; gotUSF = true; }
            else if (scene->mMetaData->Get("UnitScaleFactor", fv))
                { unitScaleFactor = fv; gotUSF = true; }
            else if (scene->mMetaData->Get("UnitScaleFactor", iv))
                { unitScaleFactor = iv; gotUSF = true; }
        }

        if (gotUSF)
            std::cout << "  FBX UnitScaleFactor=" << unitScaleFactor << "\n";
        else
            std::cout << "  FBX UnitScaleFactor not found in metadata\n";

        // UnitScaleFactor is cm/unit.  Divide by 100 to get metres/unit.
        if (gotUSF && unitScaleFactor > 0.0)
            meshScale = static_cast<float>(unitScaleFactor / 100.0);

        if (meshScale != 1.0f)
            std::cout << "  Auto-detected mesh scale (cm→m): " << meshScale << "\n";
        else
            std::cout << "  Mesh appears to already be in metres (scale=1.0)\n";
    }

    aiMatrix4x4 identity;
    collectCameraNodes   (scene, scene->mRootNode, identity, out.cameraPoses);
    collectMeshEdges     (scene, scene->mRootNode, identity, out.meshEdges);
    collectMeshTriangles (scene, scene->mRootNode, identity, out.meshTriangles);
    resolveTexturePath(scene, path, out);

    // ── Print mesh bounding box ───────────────────────────────────────────
    if (!out.meshTriangles.empty()) {
        glm::vec3 bmin( 1e30f), bmax(-1e30f);
        for (const auto& v : out.meshTriangles) {
            bmin = glm::min(bmin, v.pos);
            bmax = glm::max(bmax, v.pos);
        }
        glm::vec3 sz = bmax - bmin;
        std::cout << "  Mesh AABB (before scale): size "
                  << sz.x << " x " << sz.y << " x " << sz.z << "\n";
    }

    // ── Apply scale to mesh AND camera positions (keeps them aligned) ────
    if (meshScale != 1.0f) {
        for (auto& v : out.meshTriangles) v.pos *= meshScale;
        for (auto& v : out.meshEdges)     v.pos *= meshScale;
        for (auto& cp : out.cameraPoses)
            cp.transform[3] = glm::vec4(glm::vec3(cp.transform[3]) * meshScale, 1.f);
        std::cout << "  Applied scale " << meshScale << " to mesh and cameras.\n";
    }

    std::cout << "Loaded " << out.cameraPoses.size() << " camera poses, "
              << scene->mNumMeshes << " meshes, "
              << out.meshTriangles.size() / 3 << " triangles"
              << " (effective scale=" << meshScale << ").\n";
    return true;
}
