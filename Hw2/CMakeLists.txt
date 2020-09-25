cmake_minimum_required(VERSION 3.0.0)
project(mainProject VERSION 0.1.0)

set(CMAKE_CXX_FLAGS_DEBUG_INIT "-Wall")

include(CTest)
enable_testing()

find_package( Threads REQUIRED )
set(SOURCES ArgumentsCheck.cpp ArgumentsCheck.h
            Timer.cpp Timer.h)

add_executable(MatrixAddition MatrixAddition.cpp ${SOURCES})
target_link_libraries( MatrixAddition ${CMAKE_THREAD_LIBS_INIT} )

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
