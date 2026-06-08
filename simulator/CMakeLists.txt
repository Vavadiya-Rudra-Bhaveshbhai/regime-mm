cmake_minimum_required(VERSION 3.16)
project(regime_mm_sim CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Compiler flags ────────────────────────────────────────────────────
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -march=native -DNDEBUG)
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-g -O0 -Wall -Wextra -fsanitize=address)
    add_link_options(-fsanitize=address)
endif()

# ── Main simulator executable ─────────────────────────────────────────
add_executable(regime_mm_sim
    src/main.cpp
)

target_include_directories(regime_mm_sim PRIVATE include)

# ── Tests (optional, requires a test runner) ─────────────────────────
# add_executable(test_simulator tests/test_simulator.cpp)
# target_include_directories(test_simulator PRIVATE include)
