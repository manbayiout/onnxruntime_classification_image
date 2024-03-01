#include <iostream>
#include <opencv2/opencv.hpp>
#include <queue>
#include <mutex>
#include <thread>
#include <future>
#include <chrono>
#include <ctime>
#include <fstream>
#include <string>
#include <unistd.h>
#include <sys/wait.h>
#include <onnxruntime_cxx_api.h>
#include <cstdlib>
#include <mutex>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/sem.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <semaphore.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <hiredis/hiredis.h>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/sync/named_mutex.hpp>
#include <boost/atomic.hpp>
#include <boost/filesystem.hpp>
#include <boost/asio.hpp>


using namespace cv;
using namespace std;
using namespace Ort;
using namespace cv::dnn;
using namespace boost::interprocess;
namespace fs = boost::filesystem;


// Redis连接信息
const char* redis_host = "127.0.0.1";
const int redis_port = 6379;
const char* resultKey = "result"; //检测结果
const char* tempretureKey = "result"; //检测结果
const char* humidityKey = "result"; //检测结果
const char* dustKey = "result"; //检测结果
//软件参数
const int max_video_size = 600; // 设置列表的最大长度
size_t imageSize = 640 * 480 *3;  //设置共享内存图像的大小
boost::atomic<bool>* ptr_stop;  // 获取共享内存的指针，指向原子布尔变量
named_mutex mutex_stop(open_or_create, "mutex_stop");  // 在互斥锁的保护下修改原子布尔变量


void signalHandler(int signum) {
    // 如果是Ctrl+C信号(SIGINT)
    if (signum == SIGINT) {
        if (ptr_stop) {
            scoped_lock<named_mutex> lock(mutex_stop);
            *ptr_stop = true;
            lock.unlock();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5000));  // 睡眠一定时间模拟推理耗时
        
        _exit(EXIT_SUCCESS);
    }
}


void PreProcess(const Mat& image, Mat& image_blob)
	{
		Mat input;
		image.copyTo(input);

		//数据处理 标准化
		std::vector<Mat> channels, channel_p;
		split(input, channels);
		Mat R, G, B;
		B = channels.at(0);
		G = channels.at(1);
		R = channels.at(2);

		B = (B / 255. - 0.406) / 0.225;
		G = (G / 255. - 0.456) / 0.224;
		R = (R / 255. - 0.485) / 0.229;

		channel_p.push_back(R);
		channel_p.push_back(G);
		channel_p.push_back(B);

		Mat outt;
		merge(channel_p, outt);
		image_blob = outt;
}


void cameraProcess(void* ptr_img, named_mutex& mutex_img,
                long* ptr_nums, named_mutex& mutex_nums) {
    // 打开摄像头
    cv::VideoCapture cap(0, cv::CAP_V4L); 
	if (!cap.isOpened()) {
		std::cerr << "Error opening camera!" << std::endl;
	}

	// 设置摄像头图像的宽度和高度
    int newWidth = 640;
    int newHeight = 480;
    cap.set(cv::CAP_PROP_FRAME_WIDTH, newWidth);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, newHeight);
	double desired_fps = 20.0;  // 你想要设置的帧率
	cap.set(cv::CAP_PROP_FPS, desired_fps);

    bool stop = false;
    cv::Mat image;
    bool ret;
    
    try{
        while (!stop) {
            //获取当前stop信息
            scoped_lock<named_mutex> lock_stop(mutex_stop);
            stop = *ptr_stop;
            lock_stop.unlock();
            
            ret = cap.read(image);
            if(image.empty()){
                cout<<" no image in capture process"<<endl;
                continue;
            }

            scoped_lock<named_mutex> lock_img(mutex_img);
            // 将图像数据拷贝到共享内存
            std::memcpy(ptr_img, image.data, imageSize);
            lock_img.unlock();
            scoped_lock<named_mutex> lock_nums(mutex_nums);
            *ptr_nums += 1;
            lock_nums.unlock();

        }  
    }catch(std::exception& e){
        std::cerr << "Exception caught in Capture Process: " << e.what() << std::endl;
    }

    std::cerr << "Capture Process ends" << std::endl; 
}


void serialProcess(long* ptr_run, named_mutex& mutex_run) {
    // 连接redis
    redisContext* redisContext = redisConnect(redis_host, redis_port);
    if (redisContext && redisContext->err) {
        std::cerr << "Failed to connect to Redis: " << redisContext->errstr << std::endl;
        return ;
    }
    
    // 设置redis写入参数
	const char* command = "SET %s %b";
    cv::Mat image;
    bool ret;
    bool stop = false;

    boost::asio::io_service io;
    boost::asio::serial_port serial(io, "/dev/ttyUSB0");  // 替换为你的串口设备

    // 配置串口参数
    serial.set_option(boost::asio::serial_port_base::baud_rate(19200));
    serial.set_option(boost::asio::serial_port_base::character_size(8));
    serial.set_option(boost::asio::serial_port_base::parity(boost::asio::serial_port_base::parity::none));
    serial.set_option(boost::asio::serial_port_base::stop_bits(boost::asio::serial_port_base::stop_bits::one));
    
    try{
        while (!stop) {
            //获取当前stop信息
            scoped_lock<named_mutex> lock_stop(mutex_stop);
            stop = *ptr_stop;
            lock_stop.unlock();

            // 从串口读取数据
            const int max_read_length = 1;   
            char data_to_read[max_read_length];
            size_t read_length = boost::asio::read(serial, boost::asio::buffer(data_to_read, max_read_length));
            // // 输出读取到的数据
            // std::cout << "Read from serial: " << std::string(data_to_read, read_length) << std::endl;
            
            if (data_to_read[0] == 'X'){
                char data[7];
                size_t length = boost::asio::read(serial, boost::asio::buffer(data, 7));
                std::cout << "Read from serial: " << std::string(data, length) << std::endl;
                // 设置多个键值对
                const char *key_value_pairs[] = {"tempreture", (std::string(1, data[0]) + std::string(1, data[1])).c_str(), 
                                                "humidity", (std::string(1, data[2]) + std::string(1, data[3])).c_str(),
                                                "dust", (std::string(1, data[4]) + std::string(1, data[6])).c_str()};
                int num_pairs = sizeof(key_value_pairs) / sizeof(key_value_pairs[0]);
                const char *argv[num_pairs * 2 + 1];
                argv[0] = "MSET";
                for (int i = 0; i < num_pairs; ++i) {
                    argv[i * 2 + 1] = key_value_pairs[i * 2];
                    argv[i * 2 + 2] = key_value_pairs[i * 2 + 1];
                }

                // 发送 MSET 命令
                redisReply *reply = (redisReply *)redisCommandArgv(redisContext, num_pairs + 1, argv, NULL);
                if (reply == NULL) {
                    printf("Command execution error\n");

                }
            }else if (data_to_read[0] == 'C'){
                scoped_lock<named_mutex> lock_run(mutex_run);
                // 将图像数据拷贝到共享内存
                *ptr_run += 1;
                lock_run.unlock();
            }
        }  
    }catch(std::exception& e){
        std::cerr << "Exception caught in Serial Process: " << e.what() << std::endl;
    }
    redisFree(redisContext);  
    std::cerr << "Serial Process ends" << std::endl; 
}


void inferenceProcess(void* ptr_img, named_mutex& mutex_img,
                     boost::atomic<bool>* ptr_stop, named_mutex& mutex_stop,
                     boost::atomic<bool>* ptr_save, named_mutex& mutex_save,
                     long* ptr_run, named_mutex& mutex_run) {
    // 连接redis
    redisContext* redisContext = redisConnect(redis_host, redis_port);
    if (redisContext && redisContext->err) {
        std::cerr << "Failed to connect to Redis: " << redisContext->errstr << std::endl;
        return ;
    }
	// std::this_thread::sleep_for(std::chrono::milliseconds(5000));  // 睡眠一定时间模拟推理耗时
    // 在这里加入初始化图像分类模型的代码
	//environment （设置为VERBOSE（ORT_LOGGING_LEVEL_VERBOSE）时，方便控制台输出时看到是使用了cpu还是gpu执行）
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "OnnxModel");
	Ort::SessionOptions session_options;
	// 使用1个线程执行op,若想提升速度，增加线程数
	session_options.SetIntraOpNumThreads(1);
    //CUDA加速开启(由于onnxruntime的版本太高，无cuda_provider_factory.h的头文件，加速可以使用onnxruntime V1.8的版本)
	//OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
	// ORT_ENABLE_ALL: 启用所有可能的优化
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	//load  model and creat session
	const char* model_path = "/home/pi/deploy/cpp/onnxruntime_classification_image/models/suiPian_shuffleNet_448.onnx";
	Ort::Session session(env, model_path, session_options);
	// print model input layer (node names, types, shape etc.)
	Ort::AllocatorWithDefaultOptions allocator;
	//model info
	// 获得模型又多少个输入和输出，一般是指对应网络层的数目
	const char* input_name = "input";
	const char* output_name = "input";              
	// 自动获取维度数量
	auto input_dims = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	auto output_dims = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();    
	std::vector<const char*> input_names{ input_name };
	std::vector<const char*> output_names = { output_name };
	std::vector<const char*> input_node_names = { "input" };
	std::vector<const char*> output_node_names = { "output"};
	cv::Mat frame;
    bool stop = false;
    long nums = 0;
    long bad = 0;
    bool run = false;
    while (!stop) {
        //获取当前stop信息
        scoped_lock<named_mutex> lock_stop(mutex_stop);
        stop = *ptr_stop;
        lock_stop.unlock();
        clock_t startTime, endTime;
        startTime = clock();

        //查看是否有传感器触发
        scoped_lock<named_mutex> lock_run(mutex_run);
        if (*ptr_run != nums){
            nums = *ptr_run;
            lock_run.unlock();
        }else{
            lock_run.unlock();
            continue;
        }
        
        scoped_lock<named_mutex> lock_img(mutex_img);
        // 从共享内存读取图像数据
        cv::Mat image(Size(640, 480), CV_8UC3);
        std::memcpy(image.data, ptr_img, imageSize);
        // 解锁互斥锁
        lock_img.unlock();
        // cv::imwrite("src.png", frame);

        Mat det1, det2;
        resize(image, det1, Size(448, 448), INTER_AREA);
        det1.convertTo(det1, CV_32FC3);
        PreProcess(det1, det2);         //标准化处理
        Mat blob = dnn::blobFromImage(det2, 1., Size(448, 448), Scalar(0, 0, 0), false, true);

        //创建输入tensor
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        std::vector<Ort::Value> input_tensors;
        input_tensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, blob.ptr<float>(), blob.total(), input_dims.data(), input_dims.size()));
        /*cout << int(input_dims.size()) << endl;*/

        //(score model & input tensor, get back output tensor)
        auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), input_tensors.data(), input_names.size(), output_node_names.data(), output_node_names.size());

        assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

        // 获取输出(Get pointer to output tensor float values)
        float* floatarr = output_tensors[0].GetTensorMutableData<float>();     // 也可以使用output_tensors.front(); 获取list中的第一个元素变量  list.pop_front(); 删除list中的第一个位置的元素
        // 得到最可能分类输出
        Mat newarr = Mat_<double>(1, 2); //定义一个1*1000的矩阵

        for (int j = 0; j < newarr.cols; j++) //矩阵列数循环
        {
            newarr.at<double>(0, j) = floatarr[j];
        }
        /*cout << newarr.size() << endl;*/

        // vector<String> labels = readClassNames();
        Point classNumber;
        double classProb;
        Mat probMat = newarr(Rect(0, 0, 2, 1)).clone();
        Mat result = probMat.reshape(1, 1);
        minMaxLoc(result, NULL, &classProb, NULL, &classNumber);
        int classidx = classNumber.x;
        if (classidx ==1){
            scoped_lock<named_mutex> lock_save(mutex_save);
            *ptr_save = true;
            lock_save.unlock();
            bad += 1;
            redisReply *reply = (redisReply *)redisCommand(redisContext, "SET %s %d", resultKey, bad);
            if (reply == NULL) {
                printf("Error in SET command: %s\n", redisContext->errstr);
            }
            printf("SET command reply: %s\n", reply->str);
            freeReplyObject(reply);
        }
        // printf("\n current image classification : %s, possible : %.2f\n", labels.at(classidx).c_str(), classProb);
        // printf("\n current image classification : %d, possible : %.2f\n", classidx, classProb);
        endTime = clock();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
         std::cout << "The Inference time is:" << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
        // spdlog::info("The Inference time is:{}s", (double)(endTime - startTime) / CLOCKS_PER_SEC);
    }
    redisFree(redisContext);
    std::cerr << "Inference Process ends" << std::endl;
}

void saveVideoProcess(void* ptr_img, named_mutex& mutex_img,
                     boost::atomic<bool>* ptr_stop, named_mutex& mutex_stop,
                     boost::atomic<bool>* ptr_save, named_mutex& mutex_save,
                     long* ptr_nums, named_mutex& mutex_nums) {
    // 连接redis
    redisContext* redisContext = redisConnect(redis_host, redis_port);
    if (redisContext && redisContext->err) {
        std::cerr << "Failed to connect to Redis: " << redisContext->errstr << std::endl;
        return ;
    }
    // 定义视频编码器和写入对象
    int fourcc = cv::VideoWriter::fourcc('M', 'P', '4', 'V');
    cv::Size frameSize(640, 480);  // 你需要根据相机数据的分辨率调整这里 
    bool stop = false;
    long nums = 0;
    std::vector<cv::Mat> imageVector;
    // std::this_thread::sleep_for(std::chrono::milliseconds(8000));  // 睡眠一定时间模拟推理耗时
    try{
        while (!stop) {
            //获取当前stop信息
            scoped_lock<named_mutex> lock_stop(mutex_stop);
            stop = *ptr_stop;
            lock_stop.unlock();

            //查看图像是否更新
            scoped_lock<named_mutex> lock_nums(mutex_nums);
            if (*ptr_nums != nums){
                nums = *ptr_nums;
                lock_nums.unlock();
            }else{
                lock_nums.unlock();
                continue;
            }

            // 读取数据
            scoped_lock<named_mutex> lock_img(mutex_img);
            cv::Mat image(Size(640, 480), CV_8UC3);
            std::memcpy(image.data, ptr_img, imageSize);
            lock_img.unlock();
            imageVector.push_back(image);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            if (imageVector.size() <= max_video_size){
                continue;
            }
            image.pop_back();

            //获取当前save信息
            scoped_lock<named_mutex> lock_save(mutex_save);
            if (*ptr_save == false){
                lock_save.unlock();
                continue;
            }else{
                *ptr_save = false;
                lock_save.unlock();
            }
            
            //保存视频
            std::cout << "recording:" <<endl;
            auto now = std::chrono::system_clock::now();
            std::time_t currentTime = std::chrono::system_clock::to_time_t(now);
            std::ostringstream oss;
            oss << std::put_time(std::localtime(&currentTime), "%Y-%m-%d_%H-%M-%S") << ".mp4";
            std::string videoFileName = oss.str();
            cv::VideoWriter videoWriter(videoFileName, fourcc, 20, frameSize);
            for (const auto& originalMat : imageVector) {
                videoWriter.write(originalMat);
            }
            videoWriter.release();
            std::cout << "recording finished" <<endl;

            std::this_thread::sleep_for(std::chrono::milliseconds(5000));
        
        }
    }catch(std::exception& e){
        std::cerr << "Exception caught in Capture Process: " << e.what() << std::endl;
    }
    redisFree(redisContext);
    std::cerr << "Save Process ends" << std::endl;
}


int main() {
    // 设置信号处理函数
    if (signal(SIGINT, signalHandler) == SIG_ERR) {
        std::cerr << "Failed to set signal handler." << std::endl;
        return 1;
    }
    // // 设置进程安全的文件日志器，将日志记录到 "logfile.txt" 文件
    // auto file_logger = spdlog::basic_logger_mt<spdlog::sinks::basic_file_sink_mt>("file_logger", "logfile.txt");

    // // 设置全局默认日志器
    // spdlog::set_default_logger(file_logger);
    // file_logger->set_level(spdlog::level::trace);
        
    try{
        // 创建共享内存对象
        shared_memory_object shm_image(create_only, "shared_image", read_write);
        // 设置共享内存对象的大小
        shm_image.truncate(imageSize);
        // 映射共享内存
        mapped_region region_image(shm_image, read_write);
        // 获取共享内存的指针
        void* memPtr_image = region_image.get_address();
        // 创建互斥锁
        named_mutex mutex_image(open_or_create, "mutex_image");
        // 创建共享内存对象
        shared_memory_object shm_stop(create_only, "shared_stop", read_write);
        // 设置共享内存对象的大小
        shm_stop.truncate(sizeof(boost::atomic<bool>));
        // 映射共享内存
        mapped_region region_stop(shm_stop, read_write);
        // 获取共享内存的指针，指向原子布尔变量
        ptr_stop = new (region_stop.get_address()) boost::atomic<bool>(false);
        // 创建共享内存对象
        shared_memory_object shm_save(create_only, "shared_save", read_write);
        // 设置共享内存对象的大小
        shm_save.truncate(sizeof(boost::atomic<bool>));
        // 映射共享内存
        mapped_region region_save(shm_save, read_write);
        // 获取共享内存的指针，指向原子布尔变量
        boost::atomic<bool>* ptr_save = new (region_save.get_address()) boost::atomic<bool>(true);
        // 在互斥锁的保护下修改原子布尔变量
        named_mutex mutex_save(open_or_create, "mutex_save");
        // 创建共享内存对象
        shared_memory_object shm_nums(boost::interprocess::create_only,
            "shared_nums", boost::interprocess::read_write         
        );
        // 设置共享内存对象的大小
        shm_nums.truncate(sizeof(long));
        // 映射共享内存
        mapped_region region_nums(shm_nums, read_write);
        // 获取共享内存的指针
        long* shared_nums = static_cast<long*>(region_nums.get_address());
        // 在共享内存中写入数据
        *shared_nums = 0;
        named_mutex mutex_nums(open_or_create, "mutex_nums");
        // 创建共享内存对象
        shared_memory_object shm_run(boost::interprocess::create_only,
            "shared_run", boost::interprocess::read_write         
        );
        // 设置共享内存对象的大小
        shm_run.truncate(sizeof(long));
        // 映射共享内存
        mapped_region region_run(shm_run, read_write);
        // 获取共享内存的指针
        long* shared_run = static_cast<long*>(region_run.get_address());
        // 在共享内存中写入数据
        *shared_run = 0;
        named_mutex mutex_run(open_or_create, "mutex_run");

        pid_t childPid1 = fork();
        if (childPid1 == 0){
            cout<<"run camera capture"<<endl;
            cameraProcess(memPtr_image, mutex_image,
                        shared_nums, mutex_nums);
            return 0;
        }
        pid_t childPid2 = fork();
        if (childPid2 == 0){
            cout<<"run  serial"<<endl;
            serialProcess(shared_run, mutex_run);
            return 0;
        }
        pid_t childPid3 = fork();
        if (childPid3 == 0){
            cout<<"run  inference"<<endl;
            inferenceProcess(memPtr_image, mutex_image,
                            ptr_stop, mutex_stop,
                            ptr_save, mutex_save,
                            shared_run, mutex_run);
            return 0;
        }
        pid_t childPid4 = fork();
        if (childPid4 == 0){
        	cout<<"run video save"<<endl;
        	saveVideoProcess(memPtr_image, mutex_image,
                            ptr_stop, mutex_stop,
                            ptr_save, mutex_save,
                            shared_nums, mutex_nums);
        	return 0;
        }

        int status;
        waitpid(childPid1, &status, 0);
        waitpid(childPid2, &status, 0);
        waitpid(childPid3, &status, 0);
        waitpid(childPid4, &status, 0);

    }catch (const boost::interprocess::interprocess_exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        if (shared_memory_object::remove("shared_image")) {
            std::cout << "Shared memory object successfully removed." << std::endl;
        }
        if (shared_memory_object::remove("shared_stop")) {
            std::cout << "Shared memory object successfully removed." << std::endl;
        }
        if (shared_memory_object::remove("shared_save")) {
            std::cout << "Shared memory object successfully removed." << std::endl;
        }
        if (shared_memory_object::remove("shared_nums")) {
            std::cout << "Shared memory object successfully removed." << std::endl;
        }
    }catch(const std::exception& e){
        std::cerr << "Exception caught: " << e.what() << std::endl;
    }

    std::cout << "Parent process, PID: " << getpid() << " waiting for all child processes to finish." << std::endl;

    return 0;  // 父进程结束
}