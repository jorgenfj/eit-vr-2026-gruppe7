#pragma once

#include "scene_data.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <vector>

// ── Orbit camera ────────────────────────────────────────────────────────────
struct OrbitCamera {
    glm::vec3 target{0.f, 0.f, 0.f};
    float distance = 30.f;
    float yaw      = 0.f;   // radians
    float pitch    = 0.3f;  // radians
    float lastX = 0, lastY = 0;
    bool  dragging = false;
    bool  panning  = false;

    glm::mat4 viewMatrix() const;
};

// ── UI state for sliders / toggles ──────────────────────────────────────────
struct UIState {
    // Rays
    bool  showRays    = true;
    float rayColor[3] = {0.9f, 0.8f, 0.0f};  // RGB
    float rayAlpha    = 0.5f;

    // Mask outline
    bool  showOutline      = true;
    float outlineColor[3]  = {1.0f, 0.0f, 0.0f}; // RGB
    float outlineAlpha     = 1.0f;
    float outlineThickness = 3.0f;
};

// ── Renderer ────────────────────────────────────────────────────────────────

/// Initialise GLFW + GLEW + ImGui and create a window.  Returns nullptr on failure.
GLFWwindow* initWindow(int width = 1280, int height = 720,
                        const char* title = "Backproject – Camera Poses");

/// Upload line geometry to the GPU; returns (VAO, VBO, vertexCount).
struct GpuMesh {
    GLuint vao  = 0;
    GLuint vbo  = 0;
    GLsizei vertexCount = 0;
};
GpuMesh uploadLines(const std::vector<Vertex>& verts);

/// Upload textured triangle mesh to the GPU.
GpuMesh uploadTriangles(const std::vector<MeshVertex>& verts);

/// Decode an embedded compressed texture and upload to GL.  Returns texture id.
GLuint loadTextureFromMemory(const unsigned char* data, int length);
GLuint loadTextureFromFile(const std::string& path);

/// Compile + link the default colour shader (for lines).
GLuint createProgram();

/// Compile + link the textured+lit shader (for mesh triangles).
GLuint createMeshProgram();

/// All GPU handles needed for rendering.
struct RenderData {
    GpuMesh lines;
    GpuMesh triangles;
    GpuMesh rays;           ///< backprojection rays (GL_LINES)
    GpuMesh outline;        ///< crack mask outline (GL_LINES)
    GLuint  lineProg    = 0;
    GLuint  meshProg    = 0;
    GLuint  diffuseTex  = 0;
};

/// Run the main render loop (blocks until the window is closed).
void runRenderLoop(GLFWwindow* window, const RenderData& rd);

/// Render a single frame (poll events, draw, swap buffers).
/// Returns false when the window should close.
bool renderFrame(GLFWwindow* window, const RenderData& rd);

/// Clean-up GPU resources (including ImGui).
void cleanup(GLFWwindow* window, const RenderData& rd);

/// Access the global orbit camera (used by callbacks and the main loop).
OrbitCamera& orbitCamera();

/// Access the global UI state (used by the render loop and main).
UIState& uiState();
