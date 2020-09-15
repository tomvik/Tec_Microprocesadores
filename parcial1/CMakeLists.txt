cmake_minimum_required(VERSION 3.0.0)
project(main-07-09 VERSION 0.1.0)

include(CTest)
enable_testing()

FIND_PACKAGE ( Threads REQUIRED )
add_executable(main main.cpp)
TARGET_LINK_LIBRARIES ( main ${CMAKE_THREAD_LIBS_INIT} )

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
