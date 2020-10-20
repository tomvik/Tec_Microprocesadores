cmake_minimum_required(VERSION 3.10)
project(Homework_6 VERSION 1.0)

# Specify build type
set(CMAKE_BUILD_TYPE Debug)

message("[CMAKE] CMAKE_BUILD_TYPE= ${CMAKE_BUILD_TYPE}")

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)


# Set include directories
include_directories(include)

# Add the library
add_subdirectory(include/Timer)

# Create executable
add_executable(main src/main.cpp)

# Link the libraries
target_link_libraries( main PUBLIC
                            Timer )

# find and link OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries( main PUBLIC 
                            OpenMP::OpenMP_CXX )
endif()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
