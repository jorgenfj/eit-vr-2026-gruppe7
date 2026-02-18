#include "renderer.h"

#include "stb_image.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>

// ── Shader sources ──────────────────────────────────────────────────────────
static const char* vertSrc = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aCol;
uniform mat4 uMVP;
out vec3 vCol;
void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    vCol = aCol;
}
)";

static const char* fragSrc = R"(
#version 330 core
in  vec3 vCol;
out vec4 FragColor;
void main() {
    FragColor = vec4(vCol, 1.0);
}
)";

// ── Textured + lit mesh shaders ─────────────────────────────────────────────
static const char* meshVertSrc = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNorm;
layout(location = 2) in vec2 aUV;
uniform mat4 uMVP;
uniform mat4 uModel;       // identity for now (positions already in world space)
out vec3 vNorm;
out vec2 vUV;
out vec3 vWorldPos;
void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    vNorm     = aNorm;
    vUV       = aUV;
    vWorldPos = aPos;
}
)";

static const char* meshFragSrc = R"(
#version 330 core
in  vec3 vNorm;
in  vec2 vUV;
in  vec3 vWorldPos;
uniform sampler2D uDiffuse;
uniform vec3 uLightDir;   // direction TO light (normalised)
uniform vec3 uViewPos;    // camera position
out vec4 FragColor;
void main() {
    vec3 N = normalize(vNorm);
    vec3 L = normalize(uLightDir);

    // Simple Blinn-Phong
    float diff = max(dot(N, L), 0.0);
    // Make it two-sided so back-faces still get lit
    diff = max(diff, max(dot(-N, L), 0.0) * 0.6);
    float ambient = 0.25;

    vec3 texCol = texture(uDiffuse, vUV).rgb;
    vec3 color  = texCol * (ambient + diff * 0.75);
    FragColor   = vec4(color, 1.0);
}
)";

// ── Global orbit camera ─────────────────────────────────────────────────────
static OrbitCamera gCam;
static bool gShowRays = true;   // toggled with 'R'

OrbitCamera& orbitCamera() { return gCam; }

// ── OrbitCamera implementation ──────────────────────────────────────────────
glm::mat4 OrbitCamera::viewMatrix() const {
    float cx = target.x + distance * cosf(pitch) * sinf(yaw);
    float cy = target.y + distance * sinf(pitch);
    float cz = target.z + distance * cosf(pitch) * cosf(yaw);
    return glm::lookAt(glm::vec3(cx, cy, cz), target, {0, 1, 0});
}

// ── GLFW callbacks ──────────────────────────────────────────────────────────
static void keyCb(GLFWwindow* w, int key, int /*scancode*/, int action, int /*mods*/) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(w, true);
    if (key == GLFW_KEY_R && action == GLFW_PRESS) {
        gShowRays = !gShowRays;
        std::cout << "Backprojection rays: " << (gShowRays ? "ON" : "OFF") << "\n";
    }
}

static void scrollCb(GLFWwindow*, double, double yoff) {
    gCam.distance *= (yoff > 0) ? 0.9f : 1.1f;
    gCam.distance = std::clamp(gCam.distance, 0.5f, 500.f);
}

static void mouseButtonCb(GLFWwindow* w, int btn, int act, int) {
    if (btn == GLFW_MOUSE_BUTTON_LEFT) {
        gCam.dragging = (act == GLFW_PRESS);
        double x, y; glfwGetCursorPos(w, &x, &y);
        gCam.lastX = (float)x; gCam.lastY = (float)y;
    }
    if (btn == GLFW_MOUSE_BUTTON_MIDDLE || btn == GLFW_MOUSE_BUTTON_RIGHT) {
        gCam.panning = (act == GLFW_PRESS);
        double x, y; glfwGetCursorPos(w, &x, &y);
        gCam.lastX = (float)x; gCam.lastY = (float)y;
    }
}

static void cursorPosCb(GLFWwindow*, double xd, double yd) {
    float x = (float)xd, y = (float)yd;
    float dx = x - gCam.lastX, dy = y - gCam.lastY;
    gCam.lastX = x; gCam.lastY = y;

    if (gCam.dragging) {
        gCam.yaw   -= dx * 0.005f;
        gCam.pitch += dy * 0.005f;
        gCam.pitch  = std::clamp(gCam.pitch, -1.5f, 1.5f);
    }
    if (gCam.panning) {
        float scale = gCam.distance * 0.002f;
        glm::mat4 V = gCam.viewMatrix();
        glm::vec3 right = glm::vec3(V[0][0], V[1][0], V[2][0]);
        glm::vec3 up    = glm::vec3(V[0][1], V[1][1], V[2][1]);
        gCam.target -= right * dx * scale;
        gCam.target += up    * dy * scale;
    }
}

// ── Shader helpers ──────────────────────────────────────────────────────────
static GLuint compileShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    int ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char buf[512];
        glGetShaderInfoLog(s, 512, nullptr, buf);
        std::cerr << "Shader compile error:\n" << buf << "\n";
    }
    return s;
}

GLuint createProgram() {
    GLuint vs  = compileShader(GL_VERTEX_SHADER, vertSrc);
    GLuint fs  = compileShader(GL_FRAGMENT_SHADER, fragSrc);
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

GLuint createMeshProgram() {
    GLuint vs  = compileShader(GL_VERTEX_SHADER, meshVertSrc);
    GLuint fs  = compileShader(GL_FRAGMENT_SHADER, meshFragSrc);
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

// ── Window creation ─────────────────────────────────────────────────────────
GLFWwindow* initWindow(int width, int height, const char* title) {
    if (!glfwInit()) {
        std::cerr << "GLFW init failed\n";
        return nullptr;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);

    GLFWwindow* window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!window) {
        std::cerr << "Window creation failed\n";
        glfwTerminate();
        return nullptr;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    glfwSetScrollCallback(window, scrollCb);
    glfwSetMouseButtonCallback(window, mouseButtonCb);
    glfwSetCursorPosCallback(window, cursorPosCb);
    glfwSetKeyCallback(window, keyCb);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "GLEW init failed\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return nullptr;
    }

    return window;
}

// ── GPU upload ──────────────────────────────────────────────────────────────
GpuMesh uploadLines(const std::vector<Vertex>& verts) {
    GpuMesh m;
    m.vertexCount = (GLsizei)verts.size();

    glGenVertexArrays(1, &m.vao);
    glGenBuffers(1, &m.vbo);
    glBindVertexArray(m.vao);
    glBindBuffer(GL_ARRAY_BUFFER, m.vbo);
    glBufferData(GL_ARRAY_BUFFER,
                 verts.size() * sizeof(Vertex),
                 verts.data(), GL_STATIC_DRAW);

    // pos  (location = 0)
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          (void*)offsetof(Vertex, pos));
    // col  (location = 1)
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          (void*)offsetof(Vertex, col));

    return m;
}

// ── Triangle GPU upload ─────────────────────────────────────────────────────
GpuMesh uploadTriangles(const std::vector<MeshVertex>& verts) {
    GpuMesh m;
    m.vertexCount = (GLsizei)verts.size();
    if (verts.empty()) return m;

    glGenVertexArrays(1, &m.vao);
    glGenBuffers(1, &m.vbo);
    glBindVertexArray(m.vao);
    glBindBuffer(GL_ARRAY_BUFFER, m.vbo);
    glBufferData(GL_ARRAY_BUFFER,
                 verts.size() * sizeof(MeshVertex),
                 verts.data(), GL_STATIC_DRAW);

    // pos    (location = 0)
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(MeshVertex),
                          (void*)offsetof(MeshVertex, pos));
    // normal (location = 1)
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(MeshVertex),
                          (void*)offsetof(MeshVertex, normal));
    // uv     (location = 2)
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(MeshVertex),
                          (void*)offsetof(MeshVertex, uv));

    return m;
}

// ── Texture loading ─────────────────────────────────────────────────────────
GLuint loadTextureFromMemory(const unsigned char* data, int length) {
    int w, h, ch;
    unsigned char* pixels = stbi_load_from_memory(data, length, &w, &h, &ch, 4);
    if (!pixels) {
        std::cerr << "stb_image failed: " << stbi_failure_reason() << "\n";
        return 0;
    }
    std::cout << "Decoded texture: " << w << "x" << h << " (" << ch << " channels)\n";

    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    glGenerateMipmap(GL_TEXTURE_2D);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    stbi_image_free(pixels);
    return tex;
}

// ── Render loop ─────────────────────────────────────────────────────────────
void runRenderLoop(GLFWwindow* window, const RenderData& rd) {
    GLint lineMvpLoc = glGetUniformLocation(rd.lineProg, "uMVP");

    GLint meshMvpLoc   = glGetUniformLocation(rd.meshProg, "uMVP");
    GLint meshModelLoc = glGetUniformLocation(rd.meshProg, "uModel");
    GLint meshLightLoc = glGetUniformLocation(rd.meshProg, "uLightDir");
    GLint meshViewLoc  = glGetUniformLocation(rd.meshProg, "uViewPos");
    GLint meshTexLoc   = glGetUniformLocation(rd.meshProg, "uDiffuse");

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);
    glLineWidth(1.5f);
    glClearColor(0.12f, 0.12f, 0.14f, 1.f);

    glm::mat4 identity(1.f);
    glm::vec3 lightDir = glm::normalize(glm::vec3(0.4f, 1.0f, 0.6f));

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        int w, h;
        glfwGetFramebufferSize(window, &w, &h);
        if (h == 0) h = 1;
        glViewport(0, 0, w, h);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glm::mat4 proj = glm::perspective(glm::radians(55.f),
                                           (float)w / h, 0.1f, 500.f);
        glm::mat4 view = gCam.viewMatrix();
        glm::mat4 mvp  = proj * view;

        // ── Draw textured mesh triangles ────────────────────────────────
        if (rd.triangles.vertexCount > 0) {
            glUseProgram(rd.meshProg);
            glUniformMatrix4fv(meshMvpLoc,   1, GL_FALSE, glm::value_ptr(mvp));
            glUniformMatrix4fv(meshModelLoc, 1, GL_FALSE, glm::value_ptr(identity));
            glUniform3fv(meshLightLoc, 1, glm::value_ptr(lightDir));

            // Camera position for specular (from orbit camera)
            float cx = gCam.target.x + gCam.distance * cosf(gCam.pitch) * sinf(gCam.yaw);
            float cy = gCam.target.y + gCam.distance * sinf(gCam.pitch);
            float cz = gCam.target.z + gCam.distance * cosf(gCam.pitch) * cosf(gCam.yaw);
            glm::vec3 camPos(cx, cy, cz);
            glUniform3fv(meshViewLoc, 1, glm::value_ptr(camPos));

            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, rd.diffuseTex);
            glUniform1i(meshTexLoc, 0);

            glBindVertexArray(rd.triangles.vao);
            glDrawArrays(GL_TRIANGLES, 0, rd.triangles.vertexCount);
        }

        // ── Draw lines (grid, axes, frustums, trajectory) ──────────────
        glUseProgram(rd.lineProg);
        glUniformMatrix4fv(lineMvpLoc, 1, GL_FALSE, glm::value_ptr(mvp));
        glBindVertexArray(rd.lines.vao);
        glDrawArrays(GL_LINES, 0, rd.lines.vertexCount);

        // ── Draw backprojection rays (toggle with R) ────────────────────
        if (gShowRays && rd.rays.vertexCount > 0) {
            glLineWidth(2.0f);
            glBindVertexArray(rd.rays.vao);
            glDrawArrays(GL_LINES, 0, rd.rays.vertexCount);
            glLineWidth(1.5f);
        }

        glfwSwapBuffers(window);
    }
}

// ── Cleanup ─────────────────────────────────────────────────────────────────
void cleanup(GLFWwindow* window, const RenderData& rd) {
    glDeleteBuffers(1, &rd.lines.vbo);
    glDeleteVertexArrays(1, &rd.lines.vao);
    if (rd.triangles.vbo) {
        glDeleteBuffers(1, &rd.triangles.vbo);
        glDeleteVertexArrays(1, &rd.triangles.vao);
    }
    if (rd.rays.vbo) {
        glDeleteBuffers(1, &rd.rays.vbo);
        glDeleteVertexArrays(1, &rd.rays.vao);
    }
    if (rd.diffuseTex) glDeleteTextures(1, &rd.diffuseTex);
    glDeleteProgram(rd.lineProg);
    glDeleteProgram(rd.meshProg);
    glfwDestroyWindow(window);
    glfwTerminate();
}
