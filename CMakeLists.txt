cmake_minimum_required(VERSION 2.6)
project(librecaptcha2)

find_package(OpenCV REQUIRED)
find_library(EXPAT_LIBS expat)
find_library(CVCONVNET_LIBS cvconvnet)
find_library(FREEIMAGE_LIBS freeimage)

add_library(recaptcha2 SHARED librecaptcha2.cpp)
target_link_libraries(recaptcha2 ${CVCONVNET_LIBS} ${EXPAT_LIBS} ${FREEIMAGE_LIBS} ${OpenCV_LIBS})

set(SOURCE_FILES main.cpp)
add_executable(librecaptcha2_main ${SOURCE_FILES})
target_link_libraries(librecaptcha2_main recaptcha2)
