cmake_minimum_required(VERSION 3.10)
project(calibrateCamera)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)

find_package(Eigen3 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(calibrateCamera calibrateCamera.cpp)

target_link_libraries(calibrateCamera ${OpenCV_LIBS})