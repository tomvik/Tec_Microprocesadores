cmake_minimum_required(VERSION 3.5.1)
message(STATUS "CMake version: ${CMAKE_VERSION}")

project(main)

#option(USE_CUDA "Use CUDA" ON)

file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cu")

include_directories(include)

add_executable(main ${cpu_source_files})

#find_package(CUDA)
#CUDA_ADD_LIBRARY(main_gpu ${gpu_source_files})

find_package(OpenMP)

find_package( Threads REQUIRED )

target_link_libraries(main #main_gpu
                           OpenMP::OpenMP_CXX
                           ${CMAKE_THREAD_LIBS_INIT} )



#message(STATUS "USE_CUDA: ${USE_CUDA}")
#if(USE_CUDA)
#    add_definitions(-DUSE_CUDA)
#endif()