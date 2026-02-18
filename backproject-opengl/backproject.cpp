#include "backproject.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>

// ── CSV / cameras.txt parsing ───────────────────────────────────────────────

bool loadIntrinsics(const std::string& path, Intrinsics& intr)
{
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "Cannot open " << path << "\n";
        return false;
    }

    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;

        // Expected: CAMERA_ID MODEL WIDTH HEIGHT PARAMS...
        std::istringstream ss(line);
        int id;
        std::string model;
        ss >> id >> model >> intr.width >> intr.height;

        if (model == "SIMPLE_RADIAL") {
            ss >> intr.f >> intr.cx >> intr.cy >> intr.k1;
        } else if (model == "PINHOLE") {
            float fx, fy;
            ss >> fx >> fy >> intr.cx >> intr.cy;
            intr.f = fx;  // use fx as focal length
            intr.k1 = 0;
        } else {
            std::cerr << "Unsupported camera model: " << model << "\n";
            return false;
        }

        std::cout << "Intrinsics: " << model << " "
                  << intr.width << "x" << intr.height
                  << " f=" << intr.f << " cx=" << intr.cx
                  << " cy=" << intr.cy << " k1=" << intr.k1 << "\n";
        return true;
    }
    std::cerr << "No camera found in " << path << "\n";
    return false;
}

bool loadArucoDetections(const std::string& path,
                         std::vector<ArucoDetection>& dets)
{
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "Cannot open " << path << "\n";
        return false;
    }

    std::string line;
    std::getline(f, line); // skip header

    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        ArucoDetection d;
        char comma;
        ss >> d.frame >> comma;
        std::getline(ss, d.filename, ',');
        ss >> d.markerId >> comma >> d.centerX >> comma >> d.centerY;
        dets.push_back(d);
    }

    std::cout << "Loaded " << dets.size() << " ArUco detections.\n";
    return true;
}

// ── Ray-triangle intersection (Möller–Trumbore) ────────────────────────────

static bool rayTriangle(const glm::vec3& orig, const glm::vec3& dir,
                        const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2,
                        float& t)
{
    const float EPS = 1e-7f;
    glm::vec3 e1 = v1 - v0;
    glm::vec3 e2 = v2 - v0;
    glm::vec3 h  = glm::cross(dir, e2);
    float a = glm::dot(e1, h);
    if (std::fabs(a) < EPS) return false;

    float f = 1.f / a;
    glm::vec3 s = orig - v0;
    float u = f * glm::dot(s, h);
    if (u < 0.f || u > 1.f) return false;

    glm::vec3 q = glm::cross(s, e1);
    float v = f * glm::dot(dir, q);
    if (v < 0.f || u + v > 1.f) return false;

    t = f * glm::dot(e2, q);
    return t > EPS;
}

// ── Undistort pixel (SIMPLE_RADIAL) ─────────────────────────────────────────
// SIMPLE_RADIAL distortion: r_d = r * (1 + k1 * r^2)
// We need to invert that to get the undistorted normalised coords.
static glm::vec2 undistortPixel(float px, float py, const Intrinsics& intr)
{
    // Normalised distorted coords
    float xd = (px - intr.cx) / intr.f;
    float yd = (py - intr.cy) / intr.f;

    // Iterative undistortion (Newton-style)
    float xu = xd, yu = yd;
    for (int i = 0; i < 20; ++i) {
        float r2 = xu * xu + yu * yu;
        float scale = 1.f + intr.k1 * r2;
        xu = xd / scale;
        yu = yd / scale;
    }
    return {xu, yu};
}

// ── Build a name → index map for cameras ────────────────────────────────────
// FBX camera names: "_20260121_113044_455107"
// CSV filenames:     "20260121_113044_455107.jpg"
// Matching key: strip leading '_' from camera name, strip '.jpg' from CSV.

static std::string camNameToKey(const std::string& name)
{
    // Strip leading underscore(s)
    size_t start = name.find_first_not_of('_');
    if (start == std::string::npos) return name;
    return name.substr(start);
}

static std::string filenameToKey(const std::string& fn)
{
    // Strip extension
    auto dot = fn.rfind('.');
    return (dot != std::string::npos) ? fn.substr(0, dot) : fn;
}

// ── Public API ──────────────────────────────────────────────────────────────

int buildBackprojectionRays(const std::vector<ArucoDetection>& dets,
                            const Intrinsics& intr,
                            const std::vector<CameraPose>& cameras,
                            const std::vector<MeshVertex>& meshTris,
                            std::vector<Vertex>& rayLines)
{
    // Build lookup: key → camera index
    std::unordered_map<std::string, size_t> camMap;
    for (size_t i = 0; i < cameras.size(); ++i)
        camMap[camNameToKey(cameras[i].name)] = i;

    int hits = 0;
    glm::vec3 rayCol(1.f, 1.f, 0.f); // yellow

    for (const auto& det : dets) {
        std::string key = filenameToKey(det.filename);
        auto it = camMap.find(key);
        if (it == camMap.end()) {
            // No matching camera for this detection
            continue;
        }

        const CameraPose& cp = cameras[it->second];
        glm::vec3 origin = glm::vec3(cp.transform[3]);

        // Undistort the pixel to get normalised image coords
        glm::vec2 ndc = undistortPixel(det.centerX, det.centerY, intr);

        // Build camera-local ray direction.
        // FBX cameras look along local +X (cp.lookAt), up is cp.up.
        // ndc.x is horizontal offset, ndc.y is vertical offset.
        glm::mat3 R = glm::mat3(cp.transform);
        glm::vec3 fwd   = glm::normalize(R * cp.lookAt);
        glm::vec3 up    = glm::normalize(R * cp.up);
        glm::vec3 right = glm::normalize(glm::cross(fwd, up));
        // Re-orthogonalise up
        up = glm::cross(right, fwd);

        glm::vec3 dir = glm::normalize(fwd + right * ndc.x + up * (-ndc.y));

        // Ray-cast against all mesh triangles, find closest hit
        float bestT = std::numeric_limits<float>::max();
        bool found = false;

        for (size_t ti = 0; ti + 2 < meshTris.size(); ti += 3) {
            float t;
            if (rayTriangle(origin, dir,
                            meshTris[ti].pos, meshTris[ti+1].pos, meshTris[ti+2].pos,
                            t))
            {
                if (t < bestT) {
                    bestT = t;
                    found = true;
                }
            }
        }

        if (found) {
            glm::vec3 hitPt = origin + dir * bestT;
            rayLines.push_back({origin, rayCol});
            rayLines.push_back({hitPt,  rayCol});
            ++hits;
        }
    }

    std::cout << "Backprojection: " << hits << " / " << dets.size()
              << " rays hit the mesh.\n";
    return hits;
}
