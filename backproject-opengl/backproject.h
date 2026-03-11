#pragma once

#include "scene_data.h"
#include <string>
#include <vector>

/// Parse a COLMAP cameras.txt (supports PINHOLE and SIMPLE_RADIAL models).
bool loadIntrinsics(const std::string& path, Intrinsics& intr);

/// Parse a COLMAP images.txt and build a CameraPose per image.
/// The COLMAP world-to-camera transform (quaternion + translation) is inverted
/// to produce a camera-to-world 4×4 matrix stored in CameraPose::transform.
/// CameraPose::name is set to the image filename (e.g. "frame_30.png").
/// CameraPose::lookAt is set to (0,0,-1) and ::up to (0,-1,0) — COLMAP/OpenCV
/// convention: camera looks along -Z with Y pointing down.
bool loadColmapImages(const std::string& path,
                      std::vector<CameraPose>& poses);

/// Parse crack_pixels.csv  (columns: frame, filename, x, y).
bool loadCrackPixels(const std::string& path,
                     std::vector<CrackPixel>& pixels);

/// Parse aruco_detections.csv (legacy).
bool loadArucoDetections(const std::string& path,
                         std::vector<ArucoDetection>& dets);

/// For each crack pixel, find the matching camera pose by frame number,
/// unproject the pixel into a world-space ray, intersect with the mesh, and
/// emit a coloured line from the camera origin to the hit point.
/// Optionally also collects hit positions + normals for outline building.
///
/// Returns the number of rays that actually hit the mesh.
int buildCrackRays(const std::vector<CrackPixel>& pixels,
                   const Intrinsics& intr,
                   const std::vector<CameraPose>& cameras,
                   const std::vector<MeshVertex>& meshTris,
                   std::vector<Vertex>& rayLines,
                   std::vector<HitPoint>* hitPoints = nullptr);

/// Build a polygonal outline around clusters of ray hit points.
/// The outline is offset slightly along the surface normal so it sits
/// on top of / in front of the mesh.
void buildCrackOutline(const std::vector<HitPoint>& hits,
                       std::vector<Vertex>& outlineVerts,
                       const glm::vec3& color = glm::vec3(1.f, 0.f, 0.f));

/// Legacy: ArUco detection backprojection (FBX camera convention).
int buildBackprojectionRays(const std::vector<ArucoDetection>& dets,
                            const Intrinsics& intr,
                            const std::vector<CameraPose>& cameras,
                            const std::vector<MeshVertex>& meshTris,
                            std::vector<Vertex>& rayLines);
