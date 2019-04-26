cmake_minimum_required(VERSION 3.1.0)
project(bff)

set(CMAKE_DEBUG_POSTFIX "d")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# dependencies
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

include(CMakePackageConfigHelpers)

set(WITH_OPENMESH_INTERFACE false CACHE BOOL "Build with OpenMesh interface")
set(BUILD_VIEWER true CACHE BOOL "Build the viewer")

# suitesparse
# ------------------------------------------------------------------
# Detect SuiteSparse libraries:
# If not found automatically, set SuiteSparse_DIR in CMake to the
# directory where SuiteSparse was built.
# ------------------------------------------------------------------
set(SuiteSparse_USE_LAPACK_BLAS ON)
find_package(SuiteSparse QUIET NO_MODULE)  # 1st: Try to locate the *config.cmake file.
if(NOT SuiteSparse_FOUND)
	set(SuiteSparse_VERBOSE ON)
	find_package(SuiteSparse REQUIRED) # 2nd: Use FindSuiteSparse.cmake module
endif()
MESSAGE(STATUS "SuiteSparse_LIBS: ${SuiteSparse_LIBRARIES}")

# OpenBLAS
find_package(OpenBLAS 0.3.5 REQUIRED)

file(GLOB BFF_SOURCES
	"deps/rectangle-bin-pack/src/*.cpp" "deps/rectangle-bin-pack/include/*.h"
	"linear-algebra/src/*.cpp" "linear-algebra/include/*.h"
	"mesh/src/*.cpp" "mesh/include/*.h"
	"project/src/*.cpp" "project/include/*.h")

if(${WITH_OPENMESH_INTERFACE})
	find_package(OpenMesh REQUIRED)
	add_library(bff-openmesh-interface INTERFACE)
	target_link_libraries(bff-openmesh-interface INTERFACE
		bff
		${OPENMESH_LIBRARIES}
	)
	target_include_directories(bff-openmesh-interface INTERFACE
		${OPENMESH_INCLUDE_DIR}
	)
endif()

# create bff static library
add_library(bff STATIC ${BFF_SOURCES})
target_link_libraries(bff
	PUBLIC
	SuiteSparse::cholmod
	OpenBLAS::OpenBLAS
)
target_include_directories(bff
	PRIVATE
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/deps/rectangle-bin-pack/include>
	PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/linear-algebra/include>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/mesh/include>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/project/include>
	$<INSTALL_INTERFACE:include>
)

# build command line app
file(GLOB COMMAND_LINE_SOURCES "command-line/*.cpp")
add_executable(bff-command-line ${COMMAND_LINE_SOURCES})
target_link_libraries(bff-command-line bff)

# build viewer app
if(${BUILD_VIEWER})
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

	file(GLOB VIEWER_SOURCES "viewer/src/*.cpp" "viewer/include/*.h")

	add_executable(bff-viewer ${VIEWER_SOURCES})
	target_link_libraries(bff-viewer bff nanogui ${NANOGUI_EXTRA_LIBS})
	target_include_directories(bff-viewer PRIVATE
		$<BUILD_INTERFACE:${VIEWER_INCLUDE_DIRS}>
		$<BUILD_INTERFACE:${NANOGUI_EXTRA_INCS}>
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/deps/nanogui/include>
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/deps/nanogui/ext/glad/include>
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/deps/glm>
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/viewer/include>
	)
	target_compile_definitions(bff-viewer PRIVATE ${NANOGUI_EXTRA_DEFS})
	install(TARGETS bff-viewer DESTINATION bin)
endif()

configure_package_config_file(bffConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/bffConfig.cmake INSTALL_DESTINATION ${CMAKE_INSTALL_PREFIX})

install(TARGETS bff EXPORT bff-targets LIBRARY DESTINATION lib ARCHIVE DESTINATION lib RUNTIME DESTINATION bin)
install(TARGETS bff-command-line DESTINATION bin)
if(${WITH_OPENMESH_INTERFACE})
	install(TARGETS bff-openmesh-interface EXPORT bff-targets)
endif()

install(DIRECTORY mesh/include/ DESTINATION include FILES_MATCHING PATTERN "*.h")
install(DIRECTORY linear-algebra/include/ DESTINATION include FILES_MATCHING PATTERN "*.h")
install(DIRECTORY project/include/ DESTINATION include FILES_MATCHING PATTERN "*.h")
install(DIRECTORY mesh/include/ DESTINATION include FILES_MATCHING PATTERN "*.inl")
install(DIRECTORY linear-algebra/include/ DESTINATION include FILES_MATCHING PATTERN "*.inl")
install(DIRECTORY project/include/ DESTINATION include FILES_MATCHING PATTERN "*.inl")
if(${WITH_OPENMESH_INTERFACE})
	install(DIRECTORY openmesh-interface/include/ DESTINATION include FILES_MATCHING PATTERN "*.hpp")
endif()
install(EXPORT bff-targets FILE bffTargets.cmake DESTINATION ${CMAKE_INSTALL_PREFIX})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/bffConfig.cmake DESTINATION ${CMAKE_INSTALL_PREFIX})
