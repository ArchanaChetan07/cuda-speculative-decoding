cmake_minimum_required(VERSION 3.18)
project(spec_decode_optimizer CUDA)

# Find CUDA
find_package(CUDA 12.0 REQUIRED)

# Find PyTorch
find_package(Torch REQUIRED)

# CUDA sources
set(CUDA_SOURCES
    csrc/conflict_score.cu
)

# C++ sources
set(CPP_SOURCES
    csrc/bindings.cpp
)

# Create CUDA library
add_library(conflict_score_cuda ${CUDA_SOURCES} ${CPP_SOURCES})

# Link PyTorch
target_link_libraries(conflict_score_cuda PUBLIC ${TORCH_LIBRARIES})

# CUDA flags
set_target_properties(conflict_score_cuda PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "80;86;89;90"  # H100, A100, L40S, etc.
)

# Include directories
target_include_directories(conflict_score_cuda PUBLIC
    ${TORCH_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

# Installation
install(TARGETS conflict_score_cuda
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
