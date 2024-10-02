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

function(add_cuda exe_path)
  cmake_path(RELATIVE_PATH exe_path BASE_DIRECTORY ${PROJECT_SOURCE_DIR}/src OUTPUT_VARIABLE target_name)
  cmake_path(REPLACE_EXTENSION target_name "" OUTPUT_VARIABLE target_name)
  string(REPLACE "/" "_" target_name ${target_name})

  add_executable(${target_name} ${exe_path})
  target_link_libraries(
    ${target_name}
    PRIVATE
      CUDA::cudart
  )
  target_compile_options(
    ${target_name}
    PRIVATE
      "$<${gcc_like_cxx}:-Wall;-Wextra>"
      "$<${msvc_cxx}:/Wall>"
  )

  cmake_path(GET exe_path FILENAME exe_name)
  cmake_path(REPLACE_EXTENSION exe_name "" OUTPUT_VARIABLE exe_name)
  set_target_properties(${target_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON OUTPUT_NAME ${exe_name})
endfunction()

add_subdirectory(src)
