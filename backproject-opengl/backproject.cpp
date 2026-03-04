#include "backproject.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

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
            ss >> intr.fx >> intr.cx >> intr.cy >> intr.k1;
            intr.fy = intr.fx;
        } else if (model == "PINHOLE") {
            ss >> intr.fx >> intr.fy >> intr.cx >> intr.cy;
            intr.k1 = 0;
        } else {
            std::cerr << "Unsupported camera model: " << model << "\n";
            return false;
        }

        std::cout << "Intrinsics: " << model << " "
                  << intr.width << "x" << intr.height
                  << " fx=" << intr.fx << " fy=" << intr.fy
                  << " cx=" << intr.cx << " cy=" << intr.cy
                  << " k1=" << intr.k1 << "\n";
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

// ── COLMAP images.txt loader ────────────────────────────────────────────────
// Each image is represented by TWO lines:
//   IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
//   POINTS2D[] as (X Y POINT3D_ID) ...
// QW QX QY QZ + TX TY TZ represent the world-to-camera transform.
// We invert it to get the camera-to-world matrix stored in CameraPose::transform.

bool loadColmapImages(const std::string& path, std::vector<CameraPose>& poses)
{
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "Cannot open " << path << "\n";
        return false;
    }

    std::string line;
    while (std::getline(f, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') continue;

        // First line of the image block
        std::istringstream ss(line);
        int imageId, cameraId;
        float qw, qx, qy, qz, tx, ty, tz;
        std::string name;
        ss >> imageId >> qw >> qx >> qy >> qz >> tx >> ty >> tz >> cameraId >> name;
        if (ss.fail()) continue;

        // Skip the POINTS2D line
        std::string pointsLine;
        if (!std::getline(f, pointsLine)) break;

        // Build world-to-camera rotation from quaternion (COLMAP uses w,x,y,z order)
        glm::quat q(qw, qx, qy, qz);
        glm::mat3 R = glm::mat3_cast(q);  // world-to-camera rotation

        // world-to-camera: p_cam = R * p_world + t
        // camera-to-world: p_world = R^T * (p_cam - t) = R^T * p_cam - R^T * t
        glm::mat3 Rt = glm::transpose(R);
        glm::vec3 camPos = -(Rt * glm::vec3(tx, ty, tz));  // camera centre in world

        glm::mat4 transform(1.0f);
        // Columns 0,1,2 are the world-space axes of the camera
        transform[0] = glm::vec4(Rt[0], 0.f);
        transform[1] = glm::vec4(Rt[1], 0.f);
        transform[2] = glm::vec4(Rt[2], 0.f);
        transform[3] = glm::vec4(camPos, 1.f);

        // COLMAP/OpenCV convention: camera looks along +Z with Y down.
        // We store lookAt=(0,0,1) and up=(0,-1,0) in camera-local space.
        CameraPose cp;
        cp.name      = name;
        cp.transform = transform;
        cp.lookAt    = glm::vec3(0.f, 0.f, 1.f);   // local +Z = forward
        cp.up        = glm::vec3(0.f, -1.f, 0.f);  // local -Y = up (screen up)
        poses.push_back(cp);
    }

    std::cout << "Loaded " << poses.size() << " COLMAP camera poses.\n";
    return !poses.empty();
}

// ── crack_pixels.csv loader ─────────────────────────────────────────────────

bool loadCrackPixels(const std::string& path, std::vector<CrackPixel>& pixels)
{
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "Cannot open " << path << "\n";
        return false;
    }

    std::string line;
    std::getline(f, line); // skip header: frame,filename,x,y

    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        CrackPixel p;
        char comma;
        ss >> p.frame >> comma;
        std::getline(ss, p.filename, ',');
        ss >> p.x >> comma >> p.y;
        pixels.push_back(p);
    }

    std::cout << "Loaded " << pixels.size() << " crack pixels.\n";
    return !pixels.empty();
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
    // Normalised distorted coords (separate fx/fy)
    float xd = (px - intr.cx) / intr.fx;
    float yd = (py - intr.cy) / intr.fy;

    // Iterative undistortion (Newton-style) — only needed if k1 != 0
    float xu = xd, yu = yd;
    if (intr.k1 != 0.f) {
        for (int i = 0; i < 20; ++i) {
            float r2 = xu * xu + yu * yu;
            float scale = 1.f + intr.k1 * r2;
            xu = xd / scale;
            yu = yd / scale;
        }
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

// ── Extract frame index from COLMAP image name ───────────────────────────────
// Expects names like "frame_30.png" → returns 30.
// Returns -1 if the pattern doesn't match.

static int frameIndexFromName(const std::string& name)
{
    // Find "frame_" prefix
    const std::string prefix = "frame_";
    auto pos = name.find(prefix);
    if (pos == std::string::npos) return -1;
    pos += prefix.size();
    // Parse digits until end of digits
    size_t end = pos;
    while (end < name.size() && std::isdigit((unsigned char)name[end])) ++end;
    if (end == pos) return -1;
    return std::stoi(name.substr(pos, end - pos));
}

// ── Uniform grid for accelerated ray-triangle intersection ──────────────────
// We voxelise the mesh AABB into a regular grid and store, per cell, a list of
// triangle indices whose bounding box overlaps that cell.  During ray casting
// we walk cells along the ray with a 3-D DDA and only test triangles in cells
// the ray actually passes through.
//
// With 314k triangles a grid of ~128³ cells reduces per-ray tests from 314k
// to a few hundred on average — roughly 1000× faster.

struct UniformGrid {
    glm::vec3 bmin, bmax;                // mesh AABB (slightly padded)
    glm::ivec3 res;                      // number of cells in each dimension
    glm::vec3 cellSize;                  // size of one cell
    glm::vec3 invCellSize;              // 1/cellSize for fast coord→cell
    std::vector<std::vector<int>> cells; // cells[cellIndex] = list of triangle indices

    // Flat index from 3-D cell coords
    int idx(int x, int y, int z) const { return (z * res.y + y) * res.x + x; }

    // Clamp a world-space point to a grid cell (integer coords)
    glm::ivec3 worldToCell(const glm::vec3& p) const {
        glm::vec3 local = (p - bmin) * invCellSize;
        return glm::clamp(glm::ivec3(glm::floor(local)),
                          glm::ivec3(0), res - 1);
    }
};

static UniformGrid buildGrid(const std::vector<MeshVertex>& tris)
{
    UniformGrid g;

    // 1. Compute mesh AABB
    g.bmin = glm::vec3( std::numeric_limits<float>::max());
    g.bmax = glm::vec3(-std::numeric_limits<float>::max());
    for (const auto& v : tris) {
        g.bmin = glm::min(g.bmin, v.pos);
        g.bmax = glm::max(g.bmax, v.pos);
    }
    // Pad slightly so nothing sits exactly on the boundary
    glm::vec3 pad = (g.bmax - g.bmin) * 0.001f + glm::vec3(1e-5f);
    g.bmin -= pad;
    g.bmax += pad;

    // 2. Choose resolution: ~128 cells along the longest axis
    glm::vec3 extent = g.bmax - g.bmin;
    float longest = std::max({extent.x, extent.y, extent.z});
    int maxRes = 128;
    g.res = glm::max(glm::ivec3(glm::ceil(extent / longest * (float)maxRes)),
                     glm::ivec3(1));

    g.cellSize    = extent / glm::vec3(g.res);
    g.invCellSize = 1.f / g.cellSize;

    int totalCells = g.res.x * g.res.y * g.res.z;
    g.cells.resize(totalCells);

    // 3. Insert each triangle into all cells its AABB overlaps
    int numTris = (int)tris.size() / 3;
    for (int ti = 0; ti < numTris; ++ti) {
        const glm::vec3& v0 = tris[ti * 3 + 0].pos;
        const glm::vec3& v1 = tris[ti * 3 + 1].pos;
        const glm::vec3& v2 = tris[ti * 3 + 2].pos;
        glm::vec3 tmin = glm::min(glm::min(v0, v1), v2);
        glm::vec3 tmax = glm::max(glm::max(v0, v1), v2);
        glm::ivec3 cmin = g.worldToCell(tmin);
        glm::ivec3 cmax = g.worldToCell(tmax);
        for (int z = cmin.z; z <= cmax.z; ++z)
            for (int y = cmin.y; y <= cmax.y; ++y)
                for (int x = cmin.x; x <= cmax.x; ++x)
                    g.cells[g.idx(x, y, z)].push_back(ti);
    }

    // Stats
    size_t totalEntries = 0;
    for (auto& c : g.cells) totalEntries += c.size();
    std::cout << "Grid " << g.res.x << "×" << g.res.y << "×" << g.res.z
              << " (" << totalCells << " cells, "
              << totalEntries << " tri refs, "
              << numTris << " triangles)\n";
    return g;
}

// Walk the grid with 3-D DDA and test triangles in each cell.
// Returns true if a hit is found, setting bestT to the closest hit distance.
static bool gridRayCast(const UniformGrid& g,
                        const std::vector<MeshVertex>& tris,
                        const glm::vec3& orig, const glm::vec3& dir,
                        float& bestT)
{
    // Clip ray to grid AABB — find tMin and tMax entry/exit along ray
    glm::vec3 invDir(1.f / dir.x, 1.f / dir.y, 1.f / dir.z);
    glm::vec3 t0 = (g.bmin - orig) * invDir;
    glm::vec3 t1 = (g.bmax - orig) * invDir;
    glm::vec3 tNear = glm::min(t0, t1);
    glm::vec3 tFar  = glm::max(t0, t1);
    float tMin = std::max({tNear.x, tNear.y, tNear.z, 0.f});
    float tMax = std::min({tFar.x, tFar.y, tFar.z});
    if (tMin > tMax) return false;  // ray misses AABB

    // Entry point (clamp to just inside the grid)
    glm::vec3 entry = orig + dir * (tMin + 1e-5f);
    glm::ivec3 cell = g.worldToCell(entry);

    // DDA step and next-boundary distances
    glm::ivec3 step, out;
    glm::vec3  tDelta, tNext;
    for (int a = 0; a < 3; ++a) {
        if (dir[a] > 0.f) {
            step[a]   = 1;
            out[a]    = g.res[a];
            float edge = g.bmin[a] + (cell[a] + 1) * g.cellSize[a];
            tNext[a]  = tMin + (edge - entry[a] + g.bmin[a] - g.bmin[a]) / dir[a];
            // Simpler: distance to next cell boundary along this axis
            tNext[a]  = (g.bmin[a] + (cell[a] + 1) * g.cellSize[a] - orig[a]) * invDir[a];
            tDelta[a] = g.cellSize[a] * std::fabs(invDir[a]);
        } else if (dir[a] < 0.f) {
            step[a]   = -1;
            out[a]    = -1;
            tNext[a]  = (g.bmin[a] + cell[a] * g.cellSize[a] - orig[a]) * invDir[a];
            tDelta[a] = g.cellSize[a] * std::fabs(invDir[a]);
        } else {
            step[a]   = 0;
            out[a]    = -1;       // never exit along this axis
            tNext[a]  = std::numeric_limits<float>::max();
            tDelta[a] = std::numeric_limits<float>::max();
        }
    }

    bestT = std::numeric_limits<float>::max();
    bool found = false;

    // Walk through cells
    while (true) {
        // Test all triangles in this cell
        const auto& bucket = g.cells[g.idx(cell.x, cell.y, cell.z)];
        for (int ti : bucket) {
            float t;
            if (rayTriangle(orig, dir,
                            tris[ti * 3].pos, tris[ti * 3 + 1].pos, tris[ti * 3 + 2].pos,
                            t))
            {
                if (t < bestT) { bestT = t; found = true; }
            }
        }

        // If we've found a hit and the closest hit is before the next cell
        // boundary, we can stop — no closer hit is possible.
        float nextBoundary = std::min({tNext.x, tNext.y, tNext.z});
        if (found && bestT <= nextBoundary) break;

        // Advance to next cell along the axis with the smallest tNext
        if (tNext.x <= tNext.y && tNext.x <= tNext.z) {
            cell.x += step.x;
            if (cell.x == out.x) break;
            tNext.x += tDelta.x;
        } else if (tNext.y <= tNext.z) {
            cell.y += step.y;
            if (cell.y == out.y) break;
            tNext.y += tDelta.y;
        } else {
            cell.z += step.z;
            if (cell.z == out.z) break;
            tNext.z += tDelta.z;
        }
    }
    return found;
}

// ── Public API ──────────────────────────────────────────────────────────────

int buildCrackRays(const std::vector<CrackPixel>& pixels,
                   const Intrinsics& intr,
                   const std::vector<CameraPose>& cameras,
                   const std::vector<MeshVertex>& meshTris,
                   std::vector<Vertex>& rayLines)
{
    if (meshTris.empty() || pixels.empty()) return 0;

    // Build lookup: frame index → camera index
    std::unordered_map<int, size_t> camMap;
    for (size_t i = 0; i < cameras.size(); ++i) {
        int idx = frameIndexFromName(cameras[i].name);
        if (idx >= 0) {
            camMap[idx] = i;
            std::cout << "  Camera map: frame " << idx
                      << " → cam[" << i << "] \"" << cameras[i].name << "\""
                      << " pos=(" << cameras[i].transform[3].x
                      << "," << cameras[i].transform[3].y
                      << "," << cameras[i].transform[3].z << ")\n";
        }
    }

    // Build acceleration grid
    std::cout << "Building acceleration grid...\n";
    UniformGrid grid = buildGrid(meshTris);

    int hits = 0;
    int noCamera = 0;
    glm::vec3 rayCol(1.f, 0.2f, 0.2f); // red for cracks

    for (size_t pi = 0; pi < pixels.size(); ++pi) {
        const auto& px = pixels[pi];
        auto it = camMap.find(px.frame);
        if (it == camMap.end()) {
            ++noCamera;
            continue;
        }

        const CameraPose& cp = cameras[it->second];
        glm::vec3 origin = glm::vec3(cp.transform[3]);

        // Undistort pixel → normalised camera coords
        glm::vec2 ndc = undistortPixel(px.x, px.y, intr);

        // Build world-space ray direction from camera pose
        glm::mat3 R = glm::mat3(cp.transform);
        glm::vec3 fwd   = glm::normalize(R * cp.lookAt);
        glm::vec3 up    = glm::normalize(R * cp.up);
        glm::vec3 right = glm::normalize(glm::cross(fwd, up));
        up = glm::cross(right, fwd);  // re-orthogonalise

        glm::vec3 dir = glm::normalize(fwd + right * ndc.x + up * (-ndc.y));

        // Ray-cast through the uniform grid
        float bestT;
        if (gridRayCast(grid, meshTris, origin, dir, bestT)) {
            glm::vec3 hitPt = origin + dir * bestT;
            rayLines.push_back({origin, rayCol});
            rayLines.push_back({hitPt,  rayCol});
            ++hits;
        }

        // Progress every 5000 pixels
        if ((pi + 1) % 5000 == 0 || pi + 1 == pixels.size())
            std::cout << "  Ray progress: " << (pi + 1) << " / " << pixels.size()
                      << " (" << hits << " hits)\n";
    }

    if (noCamera > 0)
        std::cout << "Crack backprojection: " << noCamera
                  << " pixels had no matching camera.\n";
    std::cout << "Crack backprojection: " << hits << " / " << pixels.size()
              << " rays hit the mesh.\n";

    return hits;
}

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
