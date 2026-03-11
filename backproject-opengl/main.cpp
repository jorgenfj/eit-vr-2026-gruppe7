// ============================================================================
// backproject-opengl: Visualise COLMAP camera poses in an OpenGL window
// ============================================================================
#include "fbx_loader.h"
#include "fbx_exporter.h"
#include "scene_geometry.h"
#include "backproject.h"
#include "renderer.h"

#include <glm/glm.hpp>
#include <future>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <cctype>

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <resources-dir>\n";
        return 1;
    }

    std::string resDir = argv[1];
    if (!resDir.empty() && resDir.back() != '/') resDir += '/';

    // Optional second argument: mesh scale factor (default 1.0).
    float meshScale = 1.0f;
    if (argc >= 3) {
        try { meshScale = std::stof(argv[2]); }
        catch (...) { std::cerr << "Invalid mesh scale '" << argv[2] << "', using 1.0\n"; }
    }
    std::cout << "Mesh scale: " << meshScale << "\n";

    // ── Find the .fbx file in the resources directory ─────────────────────
    namespace fs = std::filesystem;
    std::string fbxPath;
    std::string texturePath;
    for (auto& p : fs::directory_iterator(resDir)) {
        if (!p.is_regular_file()) continue;
        auto ext = p.path().extension().string();
        for (auto& c : ext) c = std::tolower((unsigned char)c);
        // Skip our own exported file so we always load the original
        std::string stem = p.path().stem().string();
        if (ext == ".fbx" && fbxPath.empty() && stem != "backprojected")
            fbxPath = p.path().string();
        if ((ext == ".png" || ext == ".jpg" || ext == ".jpeg") && texturePath.empty())
            texturePath = p.path().string();
    }

    if (fbxPath.empty()) {
        std::cerr << "No .fbx file found in " << resDir << "\n";
        return 1;
    }
    std::cout << "FBX:     " << fbxPath << "\n";
    if (!texturePath.empty())
        std::cout << "Texture: " << texturePath << "\n";

    // ── Load FBX (camera poses + mesh geometry) ──────────────────────────
    SceneData scene;
    if (!loadFbx(fbxPath, scene, meshScale)) {
        std::cerr << "Failed to load FBX: " << fbxPath << "\n";
        return 1;
    }

    std::vector<CameraPose>& cameraPoses = scene.cameraPoses;
    std::cout << "Loaded " << cameraPoses.size() << " camera poses from FBX.\n";

    // If the FBX didn't resolve a valid texture file, use the one we found
    if (!texturePath.empty()) {
        bool fbxTexValid = !scene.texturePath.empty()
                        && fs::is_regular_file(scene.texturePath);
        if (!fbxTexValid) {
            std::cout << "Using auto-detected texture: " << texturePath << "\n";
            scene.texturePath = texturePath;
        }
    }

    // ── Load intrinsics ───────────────────────────────────────────────────
    Intrinsics intr;
    if (!loadIntrinsics(resDir + "cameras.txt", intr))
        return 1;

    // ── Load crack pixels ─────────────────────────────────────────────────
    std::vector<CrackPixel> crackPixels;
    loadCrackPixels(resDir + "crack_pixels.csv", crackPixels);

    // ── Build line geometry (instant) ────────────────────────────────────
    std::vector<Vertex> lineVerts;
    buildAxes(lineVerts);
    buildCameraFrustums(lineVerts, cameraPoses);
    buildTrajectory(lineVerts, cameraPoses);

    // ── Centre orbit camera ───────────────────────────────────────────────
    if (!cameraPoses.empty()) {
        glm::vec3 centre(0.f);
        for (auto& cp : cameraPoses)
            centre += glm::vec3(cp.transform[3]);
        centre /= (float)cameraPoses.size();
        OrbitCamera& cam = orbitCamera();
        cam.target   = centre;
        cam.distance = 25.f;
    }

    // ── Launch ray casting in background thread ───────────────────────────
    // Ray casting over 700k triangles can take several seconds; we don't
    // want to block the window from opening.
    struct RayResult {
        std::vector<Vertex> rays;
        std::vector<HitPoint> hitPoints;
    };
    std::future<RayResult> rayFuture = std::async(
        std::launch::async,
        [&]() {
            RayResult result;
            buildCrackRays(crackPixels, intr, cameraPoses,
                           scene.meshTriangles, result.rays, &result.hitPoints);
            return result;
        });

    // ── Init window + GL ──────────────────────────────────────────────────
    GLFWwindow* window = initWindow();
    if (!window) return 1;

    RenderData rd;
    rd.lines     = uploadLines(lineVerts);
    rd.triangles = uploadTriangles(scene.meshTriangles);
    rd.rays      = uploadLines({});   // empty for now
    rd.outline   = uploadLines({});   // empty for now
    rd.lineProg  = createProgram();
    rd.meshProg  = createMeshProgram();

    if (!scene.texturePath.empty())
        rd.diffuseTex = loadTextureFromFile(scene.texturePath);

    // ── Run render loop ───────────────────────────────────────────────────
    // Once the background thread finishes we re-upload the ray VBO.
    bool raysUploaded = false;
    std::vector<Vertex> savedRayVerts;      // kept for FBX export
    std::vector<Vertex> savedOutlineVerts;  // kept for FBX export

    while (!glfwWindowShouldClose(window)) {
        // Check if ray casting has finished
        if (!raysUploaded &&
            rayFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
        {
            RayResult result = rayFuture.get();
            std::cout << "Uploading " << result.rays.size() / 2 << " ray segments to GPU.\n";

            // Free old (empty) VBOs and upload real data
            if (rd.rays.vbo) {
                glDeleteBuffers(1, &rd.rays.vbo);
                glDeleteVertexArrays(1, &rd.rays.vao);
            }
            rd.rays = uploadLines(result.rays);
            savedRayVerts = std::move(result.rays);

            // Build and upload crack outline
            std::vector<Vertex> outlineVerts;
            buildCrackOutline(result.hitPoints, outlineVerts);
            std::cout << "Uploading " << outlineVerts.size() / 2 << " outline segments to GPU.\n";
            if (rd.outline.vbo) {
                glDeleteBuffers(1, &rd.outline.vbo);
                glDeleteVertexArrays(1, &rd.outline.vao);
            }
            rd.outline = uploadLines(outlineVerts);
            savedOutlineVerts = std::move(outlineVerts);

            raysUploaded = true;
        }

        // Handle save request from UI
        UIState& ui = uiState();
        if (ui.saveRequested) {
            ui.saveRequested = false;
            if (raysUploaded) {
                std::string outPath = resDir + "backprojected.fbx";
                glm::vec3 rc(ui.rayColor[0], ui.rayColor[1], ui.rayColor[2]);
                glm::vec3 oc(ui.outlineColor[0], ui.outlineColor[1], ui.outlineColor[2]);
                exportSceneFbx(fbxPath, outPath,
                               savedRayVerts, rc, ui.rayAlpha,
                               savedOutlineVerts, oc, ui.outlineAlpha,
                               meshScale);
            } else {
                std::cout << "Cannot save yet — ray casting still in progress.\n";
            }
        }

        renderFrame(window, rd);
    }

    cleanup(window, rd);
    return 0;
}
