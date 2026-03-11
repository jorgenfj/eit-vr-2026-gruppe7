cmake_minimum_required(VERSION 3.16)
project(backproject-opengl LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(PkgConfig REQUIRED)
find_package(OpenGL REQUIRED)

pkg_check_modules(ASSIMP REQUIRED IMPORTED_TARGET assimp)
pkg_check_modules(GLFW3 REQUIRED IMPORTED_TARGET glfw3)
pkg_check_modules(GLEW  REQUIRED IMPORTED_TARGET glew)

# ── Fetch stb_image at configure time (avoids committing it) ─────────────
include(FetchContent)
FetchContent_Declare(
    stb
    URL https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
    DOWNLOAD_NO_EXTRACT TRUE
    DOWNLOAD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/stb
)
FetchContent_MakeAvailable(stb)

# ── Fetch Dear ImGui ─────────────────────────────────────────────────────
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        v1.90.4
)
FetchContent_MakeAvailable(imgui)

# Build ImGui as a static library with OpenGL3+GLFW backends
add_library(imgui_lib STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui_lib PUBLIC
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)
target_link_libraries(imgui_lib PUBLIC
    PkgConfig::GLFW3
    PkgConfig::GLEW
    OpenGL::GL
)

add_executable(backproject
    main.cpp
    fbx_loader.cpp
    scene_geometry.cpp
    renderer.cpp
    backproject.cpp
    stb_image_impl.cpp
)
target_include_directories(backproject PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/stb)
target_link_libraries(backproject PRIVATE
    PkgConfig::ASSIMP
    PkgConfig::GLFW3
    PkgConfig::GLEW
    OpenGL::GL
    imgui_lib
)
