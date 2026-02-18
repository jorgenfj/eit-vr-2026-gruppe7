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

// ── Mesh triangle collection (textured) ─────────────────────────────────────

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
            if (f.mNumIndices != 3) continue; // skip non-triangles

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

// ── Embedded texture extraction ─────────────────────────────────────────────

static void extractEmbeddedTexture(const aiScene* scene, SceneData& out)
{
    if (scene->mNumMaterials == 0) return;

    // Look for diffuse texture on the first material
    const aiMaterial* mat = scene->mMaterials[0];
    if (mat->GetTextureCount(aiTextureType_DIFFUSE) == 0) return;

    aiString texPath;
    mat->GetTexture(aiTextureType_DIFFUSE, 0, &texPath);

    const aiTexture* tex = scene->GetEmbeddedTexture(texPath.C_Str());
    if (!tex) return;

    if (tex->mHeight == 0) {
        // Compressed format (JPEG/PNG) – mWidth is byte count
        const auto* bytes = reinterpret_cast<const unsigned char*>(tex->pcData);
        out.texture.data.assign(bytes, bytes + tex->mWidth);
        out.texture.formatHint = tex->achFormatHint;
        std::cout << "  Embedded texture: " << tex->mWidth
                  << " bytes (" << tex->achFormatHint << ")\n";
    } else {
        // Uncompressed ARGB8888 – we won't handle this for now
        std::cout << "  Embedded texture is uncompressed "
                  << tex->mWidth << "x" << tex->mHeight << " (not loaded)\n";
    }
}

// ── Public API ──────────────────────────────────────────────────────────────

bool loadFbx(const std::string& path, SceneData& out)
{
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(
        path,
        aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs);

    if (!scene || (scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) || !scene->mRootNode) {
        std::cerr << "ERROR loading FBX: " << importer.GetErrorString() << "\n";
        return false;
    }

    aiMatrix4x4 identity;
    collectCameraNodes   (scene, scene->mRootNode, identity, out.cameraPoses);
    collectMeshEdges     (scene, scene->mRootNode, identity, out.meshEdges);
    collectMeshTriangles (scene, scene->mRootNode, identity, out.meshTriangles);
    extractEmbeddedTexture(scene, out);

    std::cout << "Loaded " << out.cameraPoses.size() << " camera poses, "
              << scene->mNumMeshes << " meshes, "
              << out.meshTriangles.size() / 3 << " triangles.\n";
    return true;
}
