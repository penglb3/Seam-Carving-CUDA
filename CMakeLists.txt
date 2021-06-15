cmake_minimum_required(VERSION 3.16)
project(SEAM_CARVING_CUDA LANGUAGES CXX CUDA)

file(GLOB MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu)
file(GLOB MAIN_HDR ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

if(WIN32)
    include_directories(${PROJECT_SOURCE_DIR}/include)
    link_directories(${PROJECT_SOURCE_DIR}/lib)
else()
    find_package(OpenCV REQUIRED)
    include_directories(${OPENCV_INCLUDE_DIRS})
endif()

add_executable(seamCarving ${MAIN_SRC} ${MAIN_HDR})

if(WIN32)
    target_link_libraries(seamCarving opencv_core452 opencv_highgui452 opencv_imgcodecs452 opencv_imgproc452 opencv_photo452)
else()
    target_link_libraries(seamCarving ${OpenCV_LIBS})
endif()