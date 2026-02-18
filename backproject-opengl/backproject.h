#pragma once

#include "scene_data.h"
#include <string>
#include <vector>

/// Parse a COLMAP cameras.txt (only first SIMPLE_RADIAL camera).
bool loadIntrinsics(const std::string& path, Intrinsics& intr);

/// Parse aruco_detections.csv.
bool loadArucoDetections(const std::string& path,
                         std::vector<ArucoDetection>& dets);

/// For each detection, find the matching camera, unproject the pixel into a
/// world-space ray, intersect with the mesh, and emit a yellow line from the
/// camera origin to the hit point.
///
/// Returns the number of rays that actually hit the mesh.
int buildBackprojectionRays(const std::vector<ArucoDetection>& dets,
                            const Intrinsics& intr,
                            const std::vector<CameraPose>& cameras,
                            const std::vector<MeshVertex>& meshTris,
                            std::vector<Vertex>& rayLines);
