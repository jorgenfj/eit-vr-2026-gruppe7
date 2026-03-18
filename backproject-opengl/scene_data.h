#pragma once

#include <glm/glm.hpp>
#include <string>
#include <vector>

/// A single camera pose extracted from the FBX scene.
struct CameraPose {
    std::string name;
    glm::mat4   transform;  ///< world-space 4×4 transform of the camera node
    glm::vec3   lookAt;     ///< camera-local look direction (from aiCamera::mLookAt)
    glm::vec3   up;         ///< camera-local up direction   (from aiCamera::mUp)
};

/// Simple coloured vertex used for line drawing.
struct Vertex {
    glm::vec3 pos;
    glm::vec3 col;
};

/// Outline vertex for the 3D extruded wall.
/// The shader computes:
///   displaced = pos + normal * bump * height + side * thickness * sideOff
struct OutlineVertex {
    glm::vec3 pos;       ///< base position on mesh surface
    glm::vec3 normal;    ///< surface normal (for height extrusion)
    glm::vec3 side;      ///< perpendicular to segment on mesh surface (for width)
    glm::vec3 col;
    float     height;    ///< 0.0 = on mesh surface, 1.0 = at bump tip
    float     sideOff;   ///< -1.0 or +1.0 (left/right side of ribbon)
};

/// Vertex for mesh triangles (pos + normal + UV).
struct MeshVertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec2 uv;
};

/// SIMPLE_RADIAL camera intrinsics (COLMAP convention).
struct Intrinsics {
    int   width  = 0;
    int   height = 0;
    float fx = 0;   ///< focal length x (pixels)
    float fy = 0;   ///< focal length y (pixels)
    float cx = 0;   ///< principal point x
    float cy = 0;   ///< principal point y
    float k1 = 0;   ///< radial distortion
};

/// A single ArUco detection in a frame.
struct ArucoDetection {
    int         frame;
    std::string filename;   ///< e.g. "20260121_113044_455107.jpg"
    int         markerId;
    float       centerX;    ///< pixel x
    float       centerY;    ///< pixel y
};

/// A single crack pixel detection in a frame (from crack_pixels.csv).
struct CrackPixel {
    int         frame;      ///< matches frame number in COLMAP images.txt filename (e.g. frame_23 → 23)
    std::string filename;   ///< e.g. "frame_23.png"
    float       x;          ///< pixel x
    float       y;          ///< pixel y
};

/// A ray-mesh hit point with surface normal (used to build crack outlines).
struct HitPoint {
    glm::vec3 pos;
    glm::vec3 normal;
};

/// Everything we extract from the FBX file.
struct SceneData {
    std::vector<CameraPose> cameraPoses;
    std::vector<Vertex>     meshEdges;      ///< wireframe edges (GL_LINES pairs)
    std::vector<MeshVertex> meshTriangles;  ///< filled triangles (GL_TRIANGLES)
    std::string             texturePath;    ///< path to the diffuse texture file (may be empty)
};
