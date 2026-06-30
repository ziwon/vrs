cmake_minimum_required(VERSION 3.20)

project(vrs_deepstream_worker LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0 gstreamer-video-1.0)
pkg_check_modules(GSTREAMER_BASE REQUIRED gstreamer-base-1.0)

set(DEEPSTREAM_ROOT "/opt/nvidia/deepstream/deepstream" CACHE PATH "DeepStream SDK root")
set(DEEPSTREAM_INCLUDE_DIR "${DEEPSTREAM_ROOT}/sources/includes" CACHE PATH "DeepStream include directory")
set(DEEPSTREAM_LIB_DIR "${DEEPSTREAM_ROOT}/lib" CACHE PATH "DeepStream library directory")
set(CUDA_INCLUDE_DIR "/usr/local/cuda/include" CACHE PATH "CUDA include directory")

add_library(vrs_deepstream_metadata_core STATIC src/metadata_core.cpp)
target_include_directories(vrs_deepstream_metadata_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_options(vrs_deepstream_metadata_core PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(vrs_deepstream_metadata_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(vrs-deepstream-worker src/main.cpp)

target_include_directories(
  vrs-deepstream-worker
  PRIVATE
    ${GSTREAMER_INCLUDE_DIRS}
    ${DEEPSTREAM_INCLUDE_DIR}
)

target_compile_options(
  vrs-deepstream-worker
  PRIVATE
    ${GSTREAMER_CFLAGS_OTHER}
    -Wall
    -Wextra
    -Wpedantic
)

target_link_directories(vrs-deepstream-worker PRIVATE ${GSTREAMER_LIBRARY_DIRS} ${DEEPSTREAM_LIB_DIR})
target_link_libraries(
  vrs-deepstream-worker
  PRIVATE
    vrs_deepstream_metadata_core
    ${GSTREAMER_LIBRARIES}
    nvds_meta
    nvdsgst_meta
)

add_library(nvdsinfer_custom_yoloe SHARED src/yoloe_parser.cpp)
target_include_directories(
  nvdsinfer_custom_yoloe
  SYSTEM PRIVATE ${DEEPSTREAM_INCLUDE_DIR} ${CUDA_INCLUDE_DIR}
)
target_compile_options(nvdsinfer_custom_yoloe PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-parameter)

add_library(gstvrsmeta SHARED src/gst_vrsmeta.cpp)
target_include_directories(
  gstvrsmeta
  PRIVATE
    ${GSTREAMER_INCLUDE_DIRS}
    ${GSTREAMER_BASE_INCLUDE_DIRS}
    ${DEEPSTREAM_INCLUDE_DIR}
)
target_compile_options(
  gstvrsmeta
  PRIVATE
    ${GSTREAMER_CFLAGS_OTHER}
    ${GSTREAMER_BASE_CFLAGS_OTHER}
    -Wall
    -Wextra
    -Wpedantic
)
target_compile_definitions(gstvrsmeta PRIVATE PACKAGE="vrs")
target_link_directories(gstvrsmeta PRIVATE ${GSTREAMER_LIBRARY_DIRS} ${GSTREAMER_BASE_LIBRARY_DIRS})
target_link_directories(gstvrsmeta PRIVATE ${DEEPSTREAM_LIB_DIR})
target_link_libraries(
  gstvrsmeta
  PRIVATE
    vrs_deepstream_metadata_core
    ${GSTREAMER_LIBRARIES}
    ${GSTREAMER_BASE_LIBRARIES}
    nvds_meta
    nvdsgst_meta
)

add_executable(vrs-deepstream-metadata-core-test tests/test_metadata_core.cpp)
target_link_libraries(vrs-deepstream-metadata-core-test PRIVATE vrs_deepstream_metadata_core)

install(TARGETS vrs-deepstream-worker RUNTIME DESTINATION bin)
install(TARGETS nvdsinfer_custom_yoloe LIBRARY DESTINATION lib)
install(TARGETS gstvrsmeta LIBRARY DESTINATION lib/gstreamer-1.0)
