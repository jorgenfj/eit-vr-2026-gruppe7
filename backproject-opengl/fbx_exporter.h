#pragma once

#include "scene_data.h"
#include <glm/glm.hpp>
#include <string>
#include <vector>

/// Export a combined FBX file that re-imports the original FBX (preserving
/// its mesh, cameras, materials, etc.) and adds two new objects:
///
///   "CrackRays"    – line segments as a thin-cylinder mesh, one material
///                    with color + alpha from the UI.
///   "CrackOutline" – the mask outline as a thin-ribbon mesh, one material
///                    with color + alpha from the UI.
///
/// Lines are converted to thin cylinder / ribbon geometry because FBX does
/// not natively support GL_LINES — Blender would ignore them.
///
/// @param meshScale  The scale applied when loading the FBX (e.g. 0.01 for
///                    cm→m).  Rays/outline are in scaled space; the exporter
///                    divides by meshScale to restore original FBX units.
///
/// Returns true on success.
bool exportSceneFbx(const std::string& originalFbxPath,
                    const std::string& outputPath,
                    const std::vector<Vertex>& rayVerts,       // GL_LINES pairs
                    const glm::vec3& rayColor,
                    float rayAlpha,
                    const std::vector<Vertex>& outlineVerts,   // GL_LINES pairs
                    const glm::vec3& outlineColor,
                    float outlineAlpha,
                    float meshScale = 1.0f);
