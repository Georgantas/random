
cmake_minimum_required(VERSION 3.10.0)

project(benchmark_matrix_multiply LANGUAGES CXX CUDA)

find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

find_package(OpenCL REQUIRED)

add_executable(
    ${PROJECT_NAME}
    matrix_multiply.cpp
    # matrix_multiply_kernel.cl
    # 1024_matrix_data
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDART_LIBRARY cudart)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(
    ${PROJECT_NAME}
    ${CUDART_LIBRARY}
    OpenCL::OpenCL
)

configure_file(matrix_multiply_kernel.cl matrix_multiply_kernel.cl COPYONLY)
configure_file(1024_matrix_data 1024_matrix_data COPYONLY)
