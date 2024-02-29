cmake_minimum_required(VERSION 3.10)

project(BigDefectDetection)

# 添加源文件
set(SOURCES
    ../src/main.cpp
    # 添加其他源文件
)

# 创建可执行文件
add_executable(${PROJECT_NAME} ${SOURCES})

# 查找 Boost 库
find_package(Boost COMPONENTS system thread REQUIRED)
# 添加 Boost 库的头文件目录
include_directories(${Boost_INCLUDE_DIRS})
# 添加 Boost 库的库文件目录
link_directories(${Boost_LIBRARY_DIRS})
# 链接 Boost 库
target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})

# 查找 OpenCV 库
set(OpenCV_DIR "/home/pi/opencv/build/")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

#添加onnxruntime的头文件目录和库目录
include_directories(/usr/local/include/onnxruntime/core/session/)
link_directories(/usr/local/lib)
target_link_libraries(${PROJECT_NAME} onnxruntime)

#查找hiredis库
include_directories(/home/pi/redis/deps/)
link_directories(/usr/lib/aarch64-linux-gnu/)
target_link_libraries(${PROJECT_NAME} hiredis)

#查找spdlog库
find_package(spdlog REQUIRED)
target_link_libraries(${PROJECT_NAME} spdlog::spdlog) 

# 查找并链接 serial 库
include_directories(/usr/include/libserial/)
target_link_libraries(${PROJECT_NAME} serial)

# 如果使用 C++11 或更新版本的标准，可能需要添加以下行
set(CMAKE_CXX_STANDARD 11)  # 可根据需要更改版本号
set(CMAKE_CXX_STANDARD_REQUIRED ON)
