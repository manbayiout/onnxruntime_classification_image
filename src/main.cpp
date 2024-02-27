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
#include <hiredis/hiredis.h>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/sync/named_mutex.hpp>
#include <boost/atomic.hpp>

using namespace cv;     //当定义这一行后，cv::imread可以直接写成imread
using namespace std;
using namespace Ort;
using namespace cv::dnn;
using namespace boost::interprocess;


#define SHARED_MEM_SIZE 640*480*3*1200
// Redis连接信息
const char* redis_host = "127.0.0.1";
const int redis_port = 6379;
const int max_video_size = 600; // 设置列表的最大长度
const char* listKey = "result";
size_t imageSize = 640 * 480 *3;  //设置共享内存图像的大小
boost::atomic<bool>* ptr_stop;  // 获取共享内存的指针，指向原子布尔变量
named_mutex mutex_stop(open_or_create, "mutex_stop");  // 在互斥锁的保护下修改原子布尔变量


void signalHandler(int signum) {
    // 在信号处理函数中执行共享内存的读写操作

    // 如果是Ctrl+C信号(SIGINT)
    if (signum == SIGINT) {
        if (ptr_stop) {
            scoped_lock<named_mutex> lock(mutex_stop);
            *ptr_stop = true;
            lock.unlock();
        }

        // 在信号处理函数中调用_exit，确保是异步安全的
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


void cameraProcess(void* ptr_img, named_mutex& mutex_img) {
    // 连接redis
    redisContext* redisContext = redisConnect("127.0.0.1", 6379);
    if (redisContext && redisContext->err) {
        std::cerr << "Failed to connect to Redis: " << redisContext->errstr << std::endl;
        return ;
    }

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
    
    // 设置redis写入参数
	const char* command = "SET %s %b";
    cv::Mat image;
    bool ret;
    bool stop = false;
    
    try{
        while (!stop) {
            ret = cap.read(image);
            if(image.empty()){
                cout<<" no image in capture process"<<endl;
                continue;
            }
            // 等待互斥锁
            scoped_lock<named_mutex> lock_img(mutex_img);
            // 将图像数据拷贝到共享内存
            std::memcpy(ptr_img, image.data, imageSize);
            // 解锁互斥锁
            lock_img.unlock();

            std::this_thread::sleep_for(std::chrono::milliseconds(10));  // 睡眠一定时间模拟推理耗时
        }  
    }catch(std::exception& e){
        std::cerr << "Exception caught in Capture Process: " << e.what() << std::endl;
    }
    redisFree(redisContext);   
}


void inferenceProcess(void* ptr_img, named_mutex& mutex_img,
                     boost::atomic<bool>* ptr_stop, named_mutex& mutex_stop,
                     boost::atomic<bool>* ptr_save, named_mutex& mutex_save) {
    // 连接redis
    redisContext* redisContext = redisConnect("127.0.0.1", 6379);
    if (redisContext && redisContext->err) {
        std::cerr << "Failed to connect to Redis: " << redisContext->errstr << std::endl;
        return ;
    }
	std::this_thread::sleep_for(std::chrono::milliseconds(10000));  // 睡眠一定时间模拟推理耗时
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
	const char* model_path = "/home/pi/code/onnxruntime_deploy/claasify/weights/suiPian_mobilev3Small_448.onnx";
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
	std::cout << "input_dims:" << input_dims[0] << std::endl;
	std::cout << "output_dims:" << output_dims[0] << std::endl;    
	std::vector<const char*> input_names{ input_name };
	std::vector<const char*> output_names = { output_name };
	std::vector<const char*> input_node_names = { "input" };
	std::vector<const char*> output_node_names = { "output"};
	cv::Mat frame;
    bool stop = false;
    while (!stop) {
        //获取当前stop信息
        scoped_lock<named_mutex> lock_stop(mutex_stop);
        stop = *ptr_stop;
        lock_stop.unlock();
        clock_t startTime, endTime;
        startTime = clock();
        // 等待互斥锁
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
        }
        // printf("\n current image classification : %s, possible : %.2f\n", labels.at(classidx).c_str(), classProb);
        // printf("\n current image classification : %d, possible : %.2f\n", classidx, classProb);
        endTime = clock();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        std::cout << "The Inference time is:" << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
    }
    redisFree(redisContext);
}

void saveVideoProcess(void* ptr_img, named_mutex& mutex_img,
                     boost::atomic<bool>* ptr_stop, named_mutex& mutex_stop,
                     boost::atomic<bool>* ptr_save, named_mutex& mutex_save) {
    // 连接redis
    redisContext* redisContext = redisConnect("127.0.0.1", 6379);
    if (redisContext && redisContext->err) {
        std::cerr << "Failed to connect to Redis: " << redisContext->errstr << std::endl;
        return ;
    }
    // 定义视频编码器和写入对象
    int fourcc = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
    cv::Size frameSize(640, 480);  // 你需要根据相机数据的分辨率调整这里 
    bool stop = false;
    std::vector<cv::Mat> imageVector;
    try{
        while (!stop) {
            //获取当前stop信息
            scoped_lock<named_mutex> lock_stop(mutex_stop);
            stop = *ptr_stop;
            lock_stop.unlock();

            // 读取数据
            scoped_lock<named_mutex> lock_img(mutex_img);
            cv::Mat image(Size(640, 480), CV_8UC3);
            std::memcpy(image.data, ptr_img, imageSize);
            lock_img.unlock();
            imageVector.push_back(image);
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
            auto now_c = std::chrono::system_clock::to_time_t(now);
            std::string videoFileName = "output_" + std::to_string(now_c) + ".avi";
            cv::VideoWriter videoWriter(videoFileName, fourcc, 20, frameSize);
            for (const auto& originalMat : imageVector) {
                videoWriter.write(originalMat);
            }
            videoWriter.release();
            std::cout << "recording finished:" <<endl;

            std::this_thread::sleep_for(std::chrono::milliseconds(5000));
        
        }
    }catch(std::exception& e){
        std::cerr << "Exception caught in Capture Process: " << e.what() << std::endl;
    }
    redisFree(redisContext);	
}


int main() {
    // 设置信号处理函数
    if (signal(SIGINT, signalHandler) == SIG_ERR) {
        std::cerr << "Failed to set signal handler." << std::endl;
        return 1;
    }
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

    try{
        pid_t childPid1 = fork();
        if (childPid1 == 0){
            cout<<"run camera capture"<<endl;
            cameraProcess(memPtr_image, mutex_image);
            return 0;
        }

        pid_t childPid2 = fork();
        if (childPid2 == 0){
            cout<<"run  inference"<<endl;
            inferenceProcess(memPtr_image, mutex_image,
                            ptr_stop, mutex_stop,
                            ptr_save, mutex_save);
            return 0;
        }

        pid_t childPid3 = fork();
        if (childPid3 == 0){
        	cout<<"run video save"<<endl;
        	saveVideoProcess(memPtr_image, mutex_image,
                            ptr_stop, mutex_stop,
                            ptr_save, mutex_save);
        	return 0;
        }

        int status;
        waitpid(childPid1, &status, 0);
        waitpid(childPid2, &status, 0);
        waitpid(childPid3, &status, 0);

    }
    catch(const std::exception& e){
        std::cerr << "Exception caught: " << e.what() << std::endl;

    }catch(...){
        std::cerr << "Unknow Exception: " << std::endl;
    }
    std::cout << "Parent process, PID: " << getpid() << " waiting for all child processes to finish." << std::endl;

    return 0;  // 父进程结束
}