cmake_minimum_required(VERSION 3.10)
project(conv LANGUAGES CXX CUDA)

# 设置 CUDA 的最低版本要求
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(${CMAKE_SOURCE_DIR}/include)

# 包含 CUDA 的头文件路径

# 添加可执行文件
add_executable(app main.cu kernel.cu tools.cu)

# 链接 CUDA 和 OpenCV 库
target_link_libraries(app ${CUDA_LIBRARIES} )
target_link_libraries(app ${OpenCV_LIBS} )
