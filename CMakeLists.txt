cmake_minimum_required(VERSION 3.1.0)
project(bff)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# dependencies
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

# bff lib
list(APPEND BFF_INCLUDE_DIRS "deps/nanogui/ext/eigen")
list(APPEND BFF_INCLUDE_DIRS "deps/rectangle-bin-pack/include")
list(APPEND BFF_INCLUDE_DIRS "linear-algebra/include")
list(APPEND BFF_INCLUDE_DIRS "mesh/include")
list(APPEND BFF_INCLUDE_DIRS "project/include")
include_directories(${BFF_INCLUDE_DIRS})

file(GLOB BFF_SOURCES "deps/rectangle-bin-pack/src/*.cpp" "linear-algebra/src/*.cpp" "mesh/src/*.cpp" "project/src/*.cpp")

# create bff static library
add_library(bff ${BFF_SOURCES})

# build command line app
file(GLOB COMMAND_LINE_SOURCES "command-line/*.cpp")

add_executable(bff-command-line ${COMMAND_LINE_SOURCES})
target_link_libraries(bff-command-line bff)

# nanogui
set(NANOGUI_BUILD_EXAMPLE OFF CACHE BOOL " " FORCE)
set(NANOGUI_BUILD_PYTHON  OFF CACHE BOOL " " FORCE)
set(NANOGUI_INSTALL       OFF CACHE BOOL " " FORCE)
add_subdirectory(deps/nanogui)
add_definitions(${NANOGUI_EXTRA_DEFS})

# viewer
list(APPEND VIEWER_INCLUDE_DIRS ${NANOGUI_EXTRA_INCS})
list(APPEND VIEWER_INCLUDE_DIRS "deps/nanogui/include")
list(APPEND VIEWER_INCLUDE_DIRS "deps/glm")
list(APPEND VIEWER_INCLUDE_DIRS "viewer/include")
include_directories(${VIEWER_INCLUDE_DIRS})

file(GLOB VIEWER_SOURCES "viewer/src/*.cpp")

# build viewer app
add_executable(bff-viewer ${VIEWER_SOURCES})
target_link_libraries(bff-viewer bff nanogui ${NANOGUI_EXTRA_LIBS})
