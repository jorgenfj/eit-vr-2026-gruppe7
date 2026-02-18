// ============================================================================
// backproject-opengl: Visualise FBX camera poses in an OpenGL window
// ============================================================================
#include "fbx_loader.h"
#include "scene_geometry.h"
#include "backproject.h"
#include "renderer.h"

#include <glm/glm.hpp>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <resources-dir>\n";
        return 1;
    }

    std::string resDir = argv[1];
    // Ensure trailing slash
    if (!resDir.empty() && resDir.back() != '/') resDir += '/';

    // ── Load FBX ────────────────────────────────────────────────────────
    SceneData scene;
    if (!loadFbx(resDir + "FirstReCapTest.fbx", scene))
        return 1;

    std::cout << "Loaded " << scene.cameraPoses.size() << " camera poses.\n";

    // ── Load intrinsics + detections ────────────────────────────────────
    Intrinsics intr;
    loadIntrinsics(resDir + "cameras.txt", intr);

    std::vector<ArucoDetection> dets;
    loadArucoDetections(resDir + "aruco_detections.csv", dets);

    // ── Build backprojection rays ───────────────────────────────────────
    std::vector<Vertex> rayLines;
    buildBackprojectionRays(dets, intr, scene.cameraPoses,
                            scene.meshTriangles, rayLines);

    // ── Build line geometry ─────────────────────────────────────────────
    std::vector<Vertex> lineVerts;

    buildAxes(lineVerts);
    buildCameraFrustums(lineVerts, scene.cameraPoses);
    buildTrajectory(lineVerts, scene.cameraPoses);

    // ── Centre orbit camera on mean camera position ─────────────────────
    if (!scene.cameraPoses.empty()) {
        glm::vec3 centre(0.f);
        for (auto& cp : scene.cameraPoses)
            centre += glm::vec3(cp.transform[3]);
        centre /= (float)scene.cameraPoses.size();

        OrbitCamera& cam = orbitCamera();
        cam.target   = centre;
        cam.distance = 25.f;
    }

    // ── Init window + GL ────────────────────────────────────────────────
    GLFWwindow* window = initWindow();
    if (!window) return 1;

    RenderData rd;
    rd.lines     = uploadLines(lineVerts);
    rd.triangles = uploadTriangles(scene.meshTriangles);
    rd.rays      = uploadLines(rayLines);
    rd.lineProg  = createProgram();
    rd.meshProg  = createMeshProgram();

    // Load embedded diffuse texture
    if (!scene.texture.data.empty())
        rd.diffuseTex = loadTextureFromMemory(scene.texture.data.data(),
                                               (int)scene.texture.data.size());

    // ── Run ─────────────────────────────────────────────────────────────
    runRenderLoop(window, rd);
    cleanup(window, rd);
    return 0;
}
