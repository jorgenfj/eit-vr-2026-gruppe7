#include "scene_geometry.h"

#include <glm/glm.hpp>

// ── Camera frustums ─────────────────────────────────────────────────────────

static void addFrustum(std::vector<Vertex>& verts,
                       const CameraPose& cp,
                       const glm::vec3& color,
                       float size)
{
    const glm::mat4& M = cp.transform;
    glm::vec3 o = glm::vec3(M[3]);

    glm::mat3 R   = glm::mat3(M);
    glm::vec3 fwd = glm::normalize(R * cp.lookAt) * size;
    glm::vec3 up  = glm::normalize(R * cp.up)     * size * 0.45f;
    glm::vec3 rt  = glm::normalize(glm::cross(fwd, R * cp.up)) * size * 0.6f;

    glm::vec3 tl = o + fwd - rt + up;
    glm::vec3 tr = o + fwd + rt + up;
    glm::vec3 bl = o + fwd - rt - up;
    glm::vec3 br = o + fwd + rt - up;

    auto line = [&](glm::vec3 a, glm::vec3 b) {
        verts.push_back({a, color});
        verts.push_back({b, color});
    };

    line(o, tl); line(o, tr); line(o, bl); line(o, br);
    line(tl, tr); line(tr, br); line(br, bl); line(bl, tl);
}

void buildCameraFrustums(std::vector<Vertex>& verts,
                         const std::vector<CameraPose>& poses,
                         const glm::vec3& color,
                         float size)
{
    for (const auto& cp : poses)
        addFrustum(verts, cp, color, size);
}

// ── Trajectory ──────────────────────────────────────────────────────────────

void buildTrajectory(std::vector<Vertex>& verts,
                     const std::vector<CameraPose>& poses,
                     const glm::vec3& color)
{
    for (size_t i = 0; i + 1 < poses.size(); ++i) {
        glm::vec3 a = glm::vec3(poses[i].transform[3]);
        glm::vec3 b = glm::vec3(poses[i + 1].transform[3]);
        verts.push_back({a, color});
        verts.push_back({b, color});
    }
}

// ── Grid ────────────────────────────────────────────────────────────────────

void buildGrid(std::vector<Vertex>& verts, float halfSize, float step)
{
    glm::vec3 col(0.3f, 0.3f, 0.3f);
    for (float v = -halfSize; v <= halfSize; v += step) {
        verts.push_back({{v, 0, -halfSize}, col});
        verts.push_back({{v, 0,  halfSize}, col});
        verts.push_back({{-halfSize, 0, v}, col});
        verts.push_back({{ halfSize, 0, v}, col});
    }
}

// ── Axes ────────────────────────────────────────────────────────────────────

void buildAxes(std::vector<Vertex>& verts, float len)
{
    verts.push_back({{0,0,0}, {1,0,0}}); verts.push_back({{len,0,0}, {1,0,0}});
    verts.push_back({{0,0,0}, {0,1,0}}); verts.push_back({{0,len,0}, {0,1,0}});
    verts.push_back({{0,0,0}, {0,0,1}}); verts.push_back({{0,0,len}, {0,0,1}});
}
