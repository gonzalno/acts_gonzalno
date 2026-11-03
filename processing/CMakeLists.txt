cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(processing)

find_package(Torch REQUIRED)

find_package(ROOT REQUIRED COMPONENTS Core RIO Tree Hist)

include(${ROOT_USE_FILE})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(
    processing track_states_processing.cpp
)

target_link_libraries(
    processing 
    "${TORCH_LIBRARIES}"
    ROOT::Core 
    ROOT::RIO 
    ROOT::Tree 
    ROOT::Hist)
