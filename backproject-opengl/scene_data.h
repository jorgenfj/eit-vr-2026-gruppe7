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

/// Vertex for textured mesh triangles (pos + normal + UV).
struct MeshVertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec2 uv;
};

/// Raw embedded texture data (compressed JPEG/PNG bytes from FBX).
struct EmbeddedTexture {
    std::vector<unsigned char> data;   ///< compressed image bytes
    std::string formatHint;            ///< e.g. "jpg", "png"
};

/// SIMPLE_RADIAL camera intrinsics (COLMAP convention).
struct Intrinsics {
    int   width  = 0;
    int   height = 0;
    float f  = 0;   ///< focal length (pixels)
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

/// Everything we extract from the FBX file.
struct SceneData {
    std::vector<CameraPose> cameraPoses;
    std::vector<Vertex>     meshEdges;      ///< wireframe edges (GL_LINES pairs)
    std::vector<MeshVertex> meshTriangles;  ///< filled triangles (GL_TRIANGLES)
    EmbeddedTexture         texture;        ///< diffuse texture
};
