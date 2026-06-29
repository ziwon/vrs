cmake_minimum_required(VERSION 3.20)

project(vrs_deepstream_worker LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0 gstreamer-video-1.0)

set(DEEPSTREAM_ROOT "/opt/nvidia/deepstream/deepstream" CACHE PATH "DeepStream SDK root")
set(DEEPSTREAM_INCLUDE_DIR "${DEEPSTREAM_ROOT}/sources/includes" CACHE PATH "DeepStream include directory")
set(DEEPSTREAM_LIB_DIR "${DEEPSTREAM_ROOT}/lib" CACHE PATH "DeepStream library directory")
set(CUDA_INCLUDE_DIR "/usr/local/cuda/include" CACHE PATH "CUDA include directory")

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
target_link_libraries(vrs-deepstream-worker PRIVATE ${GSTREAMER_LIBRARIES} nvds_meta nvdsgst_meta)

add_library(nvdsinfer_custom_yoloe SHARED src/yoloe_parser.cpp)
target_include_directories(
  nvdsinfer_custom_yoloe
  SYSTEM PRIVATE ${DEEPSTREAM_INCLUDE_DIR} ${CUDA_INCLUDE_DIR}
)
target_compile_options(nvdsinfer_custom_yoloe PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-parameter)

install(TARGETS vrs-deepstream-worker RUNTIME DESTINATION bin)
install(TARGETS nvdsinfer_custom_yoloe LIBRARY DESTINATION lib)
