cmake_minimum_required(VERSION 3.27.7)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(cuda_programming_guide_jp VERSION 0.1.0 LANGUAGES CXX CUDA)

# In-source build guard
if(${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_BINARY_DIR})
  message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt.")
endif()

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_OPTIMIZE_DEPENDENCIES ON)

find_package(CUDAToolkit REQUIRED)

# CUDA_SEPARABLE_COMPILATIONを使うため
if(CUDAToolkit_VERSION_MAJOR LESS 5)
  message(FATAL_ERROR "You must install CUDA >= 5.0.")
endif()

set(gcc_like_cxx "$<COMPILE_LANG_AND_ID:CXX,ARMClang,AppleClang,Clang,GNU,LCC>")
set(msvc_cxx "$<COMPILE_LANG_AND_ID:CXX,MSVC>")

function(add_cuda exe_name)
  add_executable(${exe_name} ${exe_name}.cu)
  target_link_libraries(
    ${exe_name}
    PRIVATE
      CUDA::cudart
  )
  target_compile_options(
    ${exe_name}
    PRIVATE
      "$<${gcc_like_cxx}:-Wall;-Wextra>"
      "$<${msvc_cxx}:/Wall>"
  )
  set_property(TARGET ${exe_name} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endfunction()

add_subdirectory(src)
