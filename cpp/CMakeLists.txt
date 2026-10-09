cmake_minimum_required(VERSION 3.16)
project(iris_adaptive LANGUAGES C CXX)

# Builds the C++ adaptive loop as a shared library (iris_adaptive.dll / libiris_adaptive.so) with REBOUND's IAS15
# compiled in statically from source.
#
#   cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
#   cmake --build cpp/build --config Release
#
# The REBOUND C sources must match the Python package you validate against (`python -c "import rebound;
# print(rebound.__version__)"`), otherwise IAS15 steps differ in the last bits. Default: the tag below.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 99)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(IRIS_REBOUND_TAG "5.2.2" CACHE STRING "REBOUND git tag to build IAS15 from")
set(IRIS_REBOUND_DIR "" CACHE PATH "Use this local REBOUND checkout instead of downloading one")
option(IRIS_NATIVE "Optimise for this CPU (-march=native / AVX2). Turn OFF for a binary you copy to other machines." ON)

if(IRIS_REBOUND_DIR)
  set(RB_SRC "${IRIS_REBOUND_DIR}")
else()
  include(FetchContent)
  FetchContent_Declare(rebound GIT_REPOSITORY https://github.com/hannorein/rebound.git
                       GIT_TAG ${IRIS_REBOUND_TAG} GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(rebound)
  set(RB_SRC "${rebound_SOURCE_DIR}")
endif()
if(NOT EXISTS "${RB_SRC}/src/rebound.h")
  message(FATAL_ERROR "REBOUND sources not found in ${RB_SRC}")
endif()

# --- REBOUND C sources (what setup.py compiles: every src/*.c, plus the WHFast512 assembly on x86-64 GCC/Clang) ---
file(GLOB RB_C "${RB_SRC}/src/*.c")
set(RB_SOURCES ${RB_C})
if(NOT MSVC AND NOT WIN32 AND NOT APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  enable_language(ASM)
  list(APPEND RB_SOURCES "${RB_SRC}/src/integrator_whfast512.s")
endif()

add_library(rebound_objs OBJECT ${RB_SOURCES})
target_include_directories(rebound_objs PUBLIC "${RB_SRC}/src")
target_compile_definitions(rebound_objs PUBLIC BUILDINGLIBREBOUND)
if(MSVC)
  target_compile_options(rebound_objs PRIVATE /O2 /fp:precise /wd4996 /wd4244)
  target_compile_definitions(rebound_objs PRIVATE _CRT_NONSTDC_NO_WARNINGS=1 _CRT_SECURE_NO_WARNINGS=1)
else()
  # -ffp-contract=off: same as REBOUND's own build (setup.py), so IAS15 is bit-identical to the pip package.
  target_compile_options(rebound_objs PRIVATE -O3 -ffp-contract=off -fno-math-errno -w)
  target_compile_definitions(rebound_objs PRIVATE _GNU_SOURCE)
endif()

# --- the engine ---
add_library(iris_adaptive SHARED src/iris_adaptive.cpp $<TARGET_OBJECTS:rebound_objs>)
target_include_directories(iris_adaptive PRIVATE include src "${RB_SRC}/src")
target_compile_definitions(iris_adaptive PRIVATE BUILDINGLIBREBOUND)
set_target_properties(iris_adaptive PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
if(MSVC)
  target_compile_options(iris_adaptive PRIVATE /O2 /fp:precise /EHsc)
  if(IRIS_NATIVE)
    target_compile_options(iris_adaptive PRIVATE /arch:AVX2)
  endif()
else()
  # No -ffast-math: the Kahan sums and the Richardson repairs rely on IEEE semantics.
  target_compile_options(iris_adaptive PRIVATE -O3 -fno-math-errno -fno-trapping-math -Wall -Wextra)
  if(IRIS_NATIVE)
    target_compile_options(iris_adaptive PRIVATE -march=native)
  endif()
  find_package(Threads REQUIRED)
  target_link_libraries(iris_adaptive PRIVATE Threads::Threads m)
  if(UNIX AND NOT APPLE)
    target_link_options(iris_adaptive PRIVATE -Wl,-z,noexecstack)
  endif()
endif()
