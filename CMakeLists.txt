cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(PathTracer_CUDA LANGUAGES CXX CUDA)

if ($ENV{CLION_IDE})
    include_directories("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\include")
endif ()

set(WIN32 1)

set(CMAKE_CXX_STANDARD 11)

add_library(cuda_base
        cudaMain.cu cudaMain.h
        dummy.cpp cudaHeaders.h)

target_compile_features(cuda_base PUBLIC cxx_std_11)
set_target_properties( cuda_base
        PROPERTIES CUDA_SEPARABLE_COMPILATION ON
        )



find_package(OpenGL REQUIRED)
add_subdirectory (external)
include_directories(
##        external/AntTweakBar-1.16/include/
        external/glfw-3.1.2/include/
        external/glm-0.9.7.1/
#        external/glew-1.13.0/include/
##        external/assimp-3.0.1270/include/
##        external/bullet-2.81-rev2613/src/
        .
)

set(OPENGL_RELATED_LIBS
        ${OPENGL_LIBRARY}
        glfw
#        glew32
        )
add_definitions(
        -DTW_STATIC
        -DTW_NO_LIB_PRAGMA
        -DTW_NO_DIRECT3D
        -DGLEW_STATIC
        -D_CRT_SECURE_NO_WARNINGS
)



set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_50,code=sm_50")

add_executable(PathTracer_CUDA
        main.cpp
        )

set_target_properties(
        PathTracer_CUDA
        PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
    )
target_link_libraries(PathTracer_CUDA cuda_base ${OPENGL_RELATED_LIBS})
