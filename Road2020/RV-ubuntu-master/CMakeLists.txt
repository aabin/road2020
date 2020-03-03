cmake_minimum_required(VERSION 3.15)
project(RV)

set(CMAKE_CXX_STANDARD 14)
#set( CMAKE_BUILD_TYPE Release )

#liblas
find_package(libLAS REQUIRED)
include_directories(${LIBLAS_INCLUDE_DIR})
LINK_LIBRARIES("liblas.so.2.4.0")

#ceres-solver
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

#OpenCV2.4
set(OpenCV_DIR "/usr/local/opencv2411/release/share/OpenCV")
find_package(OpenCV 2.4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#PCL1.9
find_package(PCL 1.9 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

#CGAL
find_package(CGAL REQUIRED)
include(${CGAL_USE_FILE})

#Eigen
#find_package(Eigen3 REQUIRED)
include_directories( ${EIGEN3_INCLUDE_DIR} )

#glut
#find_package(GLUT REQUIRED)
#include_directories(${GLUT_INCLUDE_DIR})

#opengl
#find_package(OpenGL REQUIRED)
#include_directories(${OPENGL_INCLUDE_DIR})

#structure tree
INCLUDE_DIRECTORIES( include )

AUX_SOURCE_DIRECTORY( src DIR_SRCS )

add_executable( road main.cpp ${DIR_SRCS} )

target_link_libraries(road
        ${LIBLAS_LIBS}
        ${CERES_LIBRARIES}
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        ${CGAL_LIBS}
#        ${EIGEN3_LIBS}
#        ${GLUT_LIBRARY}
#        ${OPENGL_LIBRARY}
        )