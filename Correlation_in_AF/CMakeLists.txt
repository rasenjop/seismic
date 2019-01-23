cmake_minimum_required(VERSION 3.10)
project(correlation_arrayfire)

set(CMAKE_CXX_STANDARD 11)

SET(CMAKE_CXX_FLAGS "-O3")

FIND_PACKAGE(ArrayFire REQUIRED VERSION 3.5.1)

add_library(correlation_af SHARED correlation_AF.cpp)

# Declare include directories
TARGET_INCLUDE_DIRECTORIES(correlation_af PUBLIC
                           ${ArrayFire_INCLUDE_DIRS})

# Declare compilation libraries
TARGET_LINK_LIBRARIES(correlation_af
        ${ArrayFire_Unified_LIBRARIES}
        ${CMAKE_DL_LIBS})
