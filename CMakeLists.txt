cmake_minimum_required(VERSION 3.0.0)
project(mainProject C CXX)

include(CTest)
enable_testing()

FIND_PACKAGE ( Threads REQUIRED )
add_executable(main.exe ./07-09/main.cpp)
TARGET_LINK_LIBRARIES ( main.exe ${CMAKE_THREAD_LIBS_INIT} )

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

cmake_minimum_required(VERSION 3.10.2)
