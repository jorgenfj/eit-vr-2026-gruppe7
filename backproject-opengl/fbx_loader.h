#pragma once

#include "scene_data.h"
#include <string>

/// Load an FBX (or any Assimp-supported) file and extract camera poses and
/// mesh wireframe edges.  Returns true on success.
bool loadFbx(const std::string& path, SceneData& out);
