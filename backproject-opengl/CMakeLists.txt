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
)
