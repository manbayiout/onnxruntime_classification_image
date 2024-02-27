cmake_minimum_required(VERSION 3.10)

project(BigDefectDetection)

# 查找 Boost 库
find_package(Boost COMPONENTS system thread REQUIRED)
# 添加 Boost 库的头文件目录
include_directories(${Boost_INCLUDE_DIRS})
# 添加 Boost 库的库文件目录
link_directories(${Boost_LIBRARY_DIRS})

# 添加源文件
set(SOURCES
    main.cpp
    # 添加其他源文件
)

# 创建可执行文件
add_executable(${PROJECT_NAME} ${SOURCES})

# 链接 ONNX 库
target_link_libraries(${PROJECT_NAME} onnxruntime)

# 链接 hiredis 库
target_link_libraries(${PROJECT_NAME} hiredis)

# 链接 spdlog 库
find_package(spdlog REQUIRED)
target_link_libraries(${PROJECT_NAME} spdlog::spdlog)

# 链接 Boost 库
target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})

# 如果使用 C++11 或更新版本的标准，可能需要添加以下行
set(CMAKE_CXX_STANDARD 11)  # 可根据需要更改版本号
set(CMAKE_CXX_STANDARD_REQUIRED ON)
