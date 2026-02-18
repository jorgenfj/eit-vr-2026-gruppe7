#pragma once

#include "scene_data.h"
#include <vector>

/// Append camera frustum wireframe lines for each pose.
void buildCameraFrustums(std::vector<Vertex>& verts,
                         const std::vector<CameraPose>& poses,
                         const glm::vec3& color = {1.f, 0.85f, 0.f},
                         float size = 0.5f);

/// Append lines connecting consecutive camera positions (trajectory).
void buildTrajectory(std::vector<Vertex>& verts,
                     const std::vector<CameraPose>& poses,
                     const glm::vec3& color = {0.2f, 0.8f, 0.2f});

/// Append a ground-plane grid on the XZ plane at y = 0.
void buildGrid(std::vector<Vertex>& verts,
               float halfSize = 30.f,
               float step = 2.f);

/// Append RGB world-axis lines at the origin.
void buildAxes(std::vector<Vertex>& verts, float length = 4.f);
