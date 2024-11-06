# TODO If compute capability < 9.0, I want to comment out the following line.
#      However, there is no way to detect it.
add_cuda(${CMAKE_CURRENT_SOURCE_DIR}/histogram.cu)
