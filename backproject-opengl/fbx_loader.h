#pragma once

#include "scene_data.h"
#include <string>

/// Load an FBX (or any Assimp-supported) file and extract camera poses and
/// mesh wireframe edges.
/// meshScale is applied uniformly to all vertex positions so you can match
/// the mesh's unit system to the COLMAP camera coordinates (metres).
/// Default 1.0 = no scaling.  Pass e.g. 0.001 to convert mm → m.
/// NOTE: meshScale is passed by reference — if auto-detection changes it
/// (e.g. from 1.0 to 0.01), the caller gets the updated value back so it
/// can be forwarded to the FBX exporter.
bool loadFbx(const std::string& path, SceneData& out, float& meshScale);
