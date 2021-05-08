#include <opencv2/core/cuda.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <iostream>
#include <io.h>
#include <direct.h>
#include "poseRecog.hpp"
#include "drawmode.hpp"

using namespace std;
using namespace cv;
using namespace cv::dnn;

const int POSE_PAIRS[20][2] =
{
	{0,1}, {1,2}, {2,3}, {3,4},         // thumb
	{0,5}, {5,6}, {6,7}, {7,8},         // index
	{0,9}, {9,10}, {10,11}, {11,12},    // middle
	{0,13}, {13,14}, {14,15}, {15,16},  // ring
	{0,17}, {17,18}, {18,19}, {19,20}   // small
};

string protoFile = "hand/pose_deploy.prototxt";
string weightsFile = "hand/pose_iter_102000.caffemodel";
string mpath = "../calibrate_matrix/";
string mname = "camera0.txt";
string dname = "camera0_dis.txt";
string vname = "";

//Point2f wp[4] = {Point2f(0,0),Point2f(2.3,0),Point2f(0,5.7),Point2f(3,7)};
//vector<Point2f>worldPosition(wp, wp + 4);
Point3d wp[6] = { Point3d(0, 0, 0), Point3d(2.3, 0, 0.2), Point3d(0, 5.7, -2),
				 Point3d(2, 7, 0), Point3d(4.7, 0, -0.7), Point3d(6.6, 0, -0.7) };
vector<Point3d>worldPosition(wp, wp + 6);

Vec4f sword[15] = { Vec4f(6.6, -1, -1, 1),  Vec4f(6.6, 3, -1, 1),  Vec4f(0, -1, -1, 1),
					Vec4f(0, 3, -1, 1), Vec4f(0, -4, -1, 1), Vec4f(0, 6, -1, 1),
					Vec4f(-2, -4, -1, 1), Vec4f(-2, 6, -1, 1), Vec4f(-2, 0, -1, 1),
					Vec4f(-2, 2, -1, 1), Vec4f(-2, 4, -1, 1), Vec4f(-12, 0, -1, 1),
					Vec4f(-12, 2, -1, 1), Vec4f(-12, 4, -1, 1), Vec4f(-16, 2, -1, 1)};
vector<Vec4f>bigsword(sword, sword + 15);

int nPoints = 22;
int mode =  0;
float thresh = 0.1;

int LoadCam(string , Mat&, int , int  );
int check_cuda();

int main(int argc, char **argv)
{
	if (argc > 1)
	{
		mname = argv[1];
		if (argc > 2)
		{
			dname = argv[2];
		}
		if (argc > 3)
		{
			vname = argv[3];
		}
	}

	if (check_cuda() == -1)
		return -1;
	VideoCapture cap;
	if (vname.length() > 0)
	{
		cap.open(vname);
	}
	else
	{
		cap.open(0);
	}
	if (!cap.isOpened())
	{
		cerr << "Unable to connect to camera" << endl;
		return 1;
	}
	
	Mat fram, frame_fil, frameCopy;
	cap >> fram;
	int frameWidth = cap.get(CAP_PROP_FRAME_WIDTH);
	int frameHeight = cap.get(CAP_PROP_FRAME_HEIGHT);
	float aspect_ratio = frameWidth / (float)frameHeight;
	int inHeight = 368;
	int inWidth = (int(aspect_ratio*inHeight) * 8) / 8;
	std::cout << "inWidth = " << inWidth << " ; inHeight = " << inHeight << endl;

	Mat mask(frameHeight, frameWidth, CV_8UC3, cv::Scalar(0, 0, 0));
	Point preFinger(Point(0, 0));

	//读取相机矩阵并且初始化矫正
	Mat cameraMatrix(Size(3, 3), CV_32F);
	Mat distCoeffs(Size(5, 1), CV_32F);
	if (LoadCam(mpath + mname, cameraMatrix, 3, 3) == -1)
	{
		cerr << "error when loading matrix" << endl;
		return 1;
	}
	if (LoadCam(mpath + dname, distCoeffs, 1, 5) == -1)
	{
		cerr << "error when loading distortion" << endl;
		return 1;
	}
	Mat frame, map1, map2, new_camera_matrix;
	Size frameSize(Size(frameWidth, frameHeight));
	new_camera_matrix = getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, frameSize, 1, frameSize, 0);
	//cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(), new_camera_matrix, frameSize, CV_16SC2, map1, map2);
	//cv::remap(fram, frame, map1, map2, cv::INTER_LINEAR);

	//初始化视频写入
	VideoWriter video("Output-Skeleton.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, Size(frameWidth, frameHeight));

	//网络初始化
	Net net = readNetFromCaffe(protoFile, weightsFile);
	net.setPreferableTarget(dnn::DNN_TARGET_CUDA);
	net.setPreferableBackend(dnn::DNN_BACKEND_CUDA);

	double t = 0;
	while (1)
	{

		double t = (double)cv::getTickCount();

		cap >> frame;
		//cv::remap(fram, frame, map1, map2, cv::INTER_LINEAR);
		//transpose(frame, frame);
		//flip(frame, frame, 1);
		//bilateralFilter(frame, frame_fil, 10, 20, 5);
		GaussianBlur(frame, frame_fil, Size(3, 3), 0);
		frameCopy = frame.clone();
		Mat inpBlob = blobFromImage(frame_fil, 1.0 / 255, Size(inWidth, inHeight), Scalar(0, 0, 0), false, false);

		net.setInput(inpBlob);

		Mat output = net.forward();

		int H = output.size[2];
		int W = output.size[3];

		// find the position of the body parts
		vector<Point> points(nPoints);
		int recognized = 1;
		for (int n = 0; n < nPoints; n++)
		{
			// Probability map of corresponding body's part.
			Mat probMap(H, W, CV_32F, output.ptr(0, n));
			resize(probMap, probMap, Size(frameWidth, frameHeight));

			Point maxLoc;
			double prob;
			minMaxLoc(probMap, 0, &prob, 0, &maxLoc);
			if (prob > thresh)
			{
				circle(frame, cv::Point((int)maxLoc.x, (int)maxLoc.y), 4, Scalar(0, 0, 255), -1);
				//cv::putText(frame, cv::format("%d", n), cv::Point((int)maxLoc.x, (int)maxLoc.y), cv::FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 0, 255), 2);
				recognized = recognized + 1;
			}
			points[n] = maxLoc;
		}

		if (recognized < points.size())
		{
			t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
			display(frame_fil, mask);
			flip(frame_fil, frame_fil, 1);
			cv::putText(frame_fil, cv::format("time taken = %.2f sec", t), cv::Point(50, 50), cv::FONT_HERSHEY_COMPLEX, .8, cv::Scalar(255, 50, 0), 2);
			cv::putText(frame_fil, cv::format("mode: %d ", mode), cv::Point(500, 50), cv::FONT_HERSHEY_COMPLEX, .8, cv::Scalar(255, 50, 0), 2);
			imshow("Output-Skeleton", frame_fil);
			video.write(frame_fil);
			cout << "Time Taken for frame = " << t << endl;
			char key = waitKey(1);
			if (key == 27)
				break;
			continue;
		}

		int mes = posejudge(points);

		cout << "the mode is" << mes << endl;

		if (mes == 2)
		{
			mode = 0;
		}
		else if (mes == 3)
		{
			mode = 1;
			clearmask(mask);
			
		}
		if (mode == 0)
		{
			if (mes == 1)
			{
				if (preFinger.x == 0 && preFinger.y == 0)
					preFinger = points[8];
				else
					drawLine(mask, preFinger, points[8]);
			}
			else if (mes == 4)
			{
				eraseLine(mask, points);
			}
			else if (mes == 0)
			{
				preFinger = Point(0, 0);
			}
		}
		else
		{
			vector<Point2d> projPosition(6);
			projPosition[0] = points[5];
			projPosition[1] = points[9];
			projPosition[2] = points[1];
			projPosition[3] = points[0];
			projPosition[4] = points[13];
			projPosition[5] = points[17];

			if (mes == 0)
			{
				clearmask(mask);
				Mat_<float> h;
				/* //单应矩阵求外参，失败
				h = findHomography(worldPosition, projPosition);
				vector<Mat> R_h, t_h, n;
				int solutions = decomposeHomographyMat(h, new_camera_matrix, R_h, t_h, n);
				cout << "========Homography========" << endl;
				for (int i = 0; i < solutions; ++i) {
					cout << "======== " << i << " ========" << endl;
					cout << "rotation" << i << " = " << endl;
					cout << R_h.at(i) << endl;
					cout << "translation" << i << " = " << endl;
					cout << t_h.at(i) << endl;
				}
				Mat R, Rf;
				hconcat(R_h.at(2), t_h.at(2), R);
				R.convertTo(Rf, CV_32F);
				Mat P = new_camera_matrix * Rf;
				*/
				Mat rvec = Mat::zeros(3, 1, CV_32FC1);
				Mat tvec = Mat::zeros(3, 1, CV_32FC1);
				solvePnP(worldPosition, projPosition, cameraMatrix, distCoeffs, rvec, tvec);
				Mat r, R, Rf;
				Rodrigues(rvec, r);
				hconcat(r, tvec, R);
				R.convertTo(Rf, CV_32F);
				Mat P = new_camera_matrix * Rf;
				
				Mat odian = P * Mat(Vec4f(0, 0, 0, 1));
				odian = odian / odian.at<float>(2, 0);
				Mat xzhou = P * Mat(Vec4f(2, 0, 0, 1));
				xzhou = xzhou / xzhou.at<float>(2, 0);
				Mat yzhou = P * Mat(Vec4f(0, 2, 0, 1));
				yzhou = yzhou / yzhou.at<float>(2, 0);
				Mat zzhou = P * Mat(Vec4f(0, 0, 2, 1));
				zzhou = zzhou / zzhou.at<float>(2, 0);
				Point oo(Point(int(odian.at<float>(0, 0)), int(odian.at<float>(1, 0))));
				Point xx(Point(int(xzhou.at<float>(0, 0)), int(xzhou.at<float>(1, 0))));
				Point yy(Point(int(yzhou.at<float>(0, 0)), int(yzhou.at<float>(1, 0))));
				Point zz(Point(int(zzhou.at<float>(0, 0)), int(zzhou.at<float>(1, 0))));
				line(mask, oo, xx, cv::Scalar(0, 0, 255),3);
				line(mask, oo, yy, cv::Scalar(0, 255, 0),3);
				line(mask, oo, zz, cv::Scalar(255, 255, 0),3);

				vector<Point> projsword(15);
				for (int i = 0; i < 15; ++i)
				{
					Mat ps = P * Mat(bigsword.at(i));
					ps = ps / ps.at<float>(2, 0);
					Point pros(Point(int(ps.at<float>(0, 0)), int(ps.at<float>(1, 0))));
					projsword[i] = pros;
				}
				drawsword(mask, projsword);
			}
			else if (mes == 4)
			{
				clearmask(mask);
			}
		}

		display(frame, mask);
		int nPairs = sizeof(POSE_PAIRS) / sizeof(POSE_PAIRS[0]);
		for (int n = 0; n < nPairs; n++)
		{
			// lookup 2 connected body/hand parts
			Point2f partA = points[POSE_PAIRS[n][0]];
			Point2f partB = points[POSE_PAIRS[n][1]];

			if (partA.x <= 0 || partA.y <= 0 || partB.x <= 0 || partB.y <= 0)
				continue;

			line(frame, partA, partB, Scalar(0, 255, 255), 2);
		}
		t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
		cout << "Time Taken for frame = " << t << endl;
		flip(frame, frame, 1);
		cv::putText(frame, cv::format("time taken = %.2f sec", t), cv::Point(50, 50), cv::FONT_HERSHEY_COMPLEX, .8, cv::Scalar(255, 50, 0), 2);
		cv::putText(frame, cv::format("mode: %d ", mode), cv::Point(500, 50), cv::FONT_HERSHEY_COMPLEX, .8, cv::Scalar(255, 50, 0), 2);
		// imshow("Output-Keypoints", frameCopy);
		imshow("Output-Skeleton", frame);
		video.write(frame);
		char key = waitKey(1);
		if (key == 27)
			break;
	}
	// When everything done, release the video capture and write object
	cap.release();
	video.release();

	return 0;
}

int LoadCam(string fileName, Mat& matData, int matRows = 0, int matCols = 0)
{
	int retVal = 0;

	// 打开文件
	ifstream inFile(fileName.c_str(), ios_base::in);
	if (!inFile.is_open())
	{
		cout << "读取文件失败" << endl;
		retVal = -1;
		return (retVal);
	}
	cout << "loading cammatrix...." << endl;
	// 载入数据
	for (int i = 0; i < matRows; ++i)
	{
		for (int j = 0; j < matCols; ++j)
		{
			float k;
			inFile >> k;
			matData.at<float>(i, j) = k;
		}
	}
	return (retVal);
}

int check_cuda()
{
	//cuda加速检查
	int num_devices = cuda::getCudaEnabledDeviceCount();
	if (num_devices <= 0)
	{
		std::cerr << "There is no device." << std::endl;
		return -1;
	}
	int enable_device_id = -1;
	for (int i = 0; i < num_devices; i++)
	{
		cuda::DeviceInfo dev_info(i);
		if (dev_info.isCompatible())
		{
			enable_device_id = i;
		}
	}
	if (enable_device_id < 0)
	{
		std::cerr << "GPU module isn't built for GPU" << std::endl;
		return -1;
	}
	cuda::setDevice(enable_device_id);
	std::cout << "GPU is ready, device ID is " << num_devices << "\n";
	return 0;
}