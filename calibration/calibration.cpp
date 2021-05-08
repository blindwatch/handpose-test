#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <iostream>
#include <io.h>
#include <direct.h>

// Defining the dimensions of checkerboard
int CHECKERBOARD[2]{ 8,13 };
float scale = 1.5;
std::string path = "./calibration_img_camera0/*.jpg";
std::string mpath = "./calibrate_matrix/";
std::string mname = "camera0.txt";
std::string dname = "camera0_dis.txt";

template<typename _T>
static void writeStream(std::ofstream& file, const _T* data, int rows, int cols, int matStep, int cn);
void saveAsText(std::string filename, cv::InputArray _src);

int main(int argc, char **argv)
{
	// Creating vector to store vectors of 3D points for each checkerboard image
	std::vector<std::vector<cv::Point3f> > objpoints;

	// Creating vector to store vectors of 2D points for each checkerboard image
	std::vector<std::vector<cv::Point2f> > imgpoints;

	// Defining the world coordinates for 3D points
	std::vector<cv::Point3f> objp;
	for (int i{ 0 }; i < CHECKERBOARD[1]; i++)
	{
		for (int j{ 0 }; j < CHECKERBOARD[0]; j++)
			objp.push_back(cv::Point3f(scale * j, scale * i, 0));
	}


	// Extracting path of individual image stored in a given directory
	std::vector<cv::String> images;
	// Path of the folder containing checkerboard images

	if (argc > 1)
	{
		std::string fpath;
		fpath = argv[1];
		path = fpath + "*.jpg";
		if (argc == 3)
		{
			mname = argv[2];
		}
		else if (argc == 4)
		{
			mname = argv[2];
			dname = argv[3];
		}
	}
	
	cv::glob(path, images);

	cv::Mat frame, gray;
	// vector to store the pixel coordinates of detected checker board corners 
	std::vector<cv::Point2f> corner_pts;
	bool success;

	// Looping over all the images in the directory
	for (int i{ 0 }; i < images.size(); i++)
	{
		frame = cv::imread(images[i]);
		cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

		// Finding checker board corners
		// If desired number of corners are found in the image then success = true  
		success = cv::findChessboardCorners(gray, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);

		/*
		 * If desired number of corner are detected,
		 * we refine the pixel coordinates and display
		 * them on the images of checker board
		*/
		if (success)
		{
			cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.001);

			// refining pixel coordinates for given 2d points.
			cv::cornerSubPix(gray, corner_pts, cv::Size(11, 11), cv::Size(-1, -1), criteria);

			// Displaying the detected corner points on the checker board
			cv::drawChessboardCorners(frame, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, success);

			objpoints.push_back(objp);
			imgpoints.push_back(corner_pts);
		}

		cv::imshow("Image", frame);
		cv::waitKey(0);
	}

	cv::destroyAllWindows();

	cv::Mat cameraMatrix, distCoeffs, R, T;

	/*
	 * Performing camera calibration by
	 * passing the value of known 3D points (objpoints)
	 * and corresponding pixel coordinates of the
	 * detected corners (imgpoints)
	*/
	cv::calibrateCamera(objpoints, imgpoints, cv::Size(gray.rows, gray.cols), cameraMatrix, distCoeffs, R, T);

	std::cout << "cameraMatrix : " << cameraMatrix << std::endl;
	std::cout << "distCoeffs : " << distCoeffs << std::endl;
	std::cout << "Rotation vector : " << R << std::endl;
	std::cout << "Translation vector : " << T << std::endl;

	if (_access(mpath.c_str(), 0) == -1)
		_mkdir(mpath.c_str());
	std::string pinmatrix = mpath + mname;
	std::string pdimatrix = mpath + dname;
	saveAsText(pinmatrix, cameraMatrix);
	saveAsText(pdimatrix, distCoeffs);

	// Trying to undistort the image using the camera parameters obtained from calibration

	cv::Mat image;
	image = cv::imread(images[0]);
	cv::Mat dst, map1, map2, new_camera_matrix;
	cv::Size imageSize(cv::Size(image.cols, image.rows));

	// Refining the camera matrix using parameters obtained by calibration
	new_camera_matrix = cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0);

	// Method 2 to undistort the image
	cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(), new_camera_matrix, imageSize, CV_16SC2, map1, map2);

	cv::remap(image, dst, map1, map2, cv::INTER_LINEAR);

	//Displaying the undistorted image
	cv::imshow("undistorted image", dst);
	std::string img_path = "./undis_img/";
	if (_access(img_path.c_str(), 0) == -1)
		_mkdir(img_path.c_str());
	cv::imwrite(img_path + "0.jpg", dst);
	cv::waitKey(0);
	
	for (int i = 1; i < images.size(); ++i)
	{
		image = cv::imread(images[i]);
		cv::remap(image, dst, map1, map2, cv::INTER_LINEAR);
		cv::imshow("undistorted image", dst);
		cv::imwrite(img_path + "undis_" + std::to_string(i) + ".jpg", dst);
		cv::waitKey(0);
	}

	return 0;
}

template<typename _T>
static void writeStream(std::ofstream& file, const _T* data, int rows, int cols, int matStep, int cn)
{
	int x, y;
	for (y = 0; y < rows; y++)
	{
		const _T* pData = data + y * matStep;
		for (x = 0; x < cols*cn; x += cn)
		{
			if (cn == 1)
			{
				file << pData[x] << "\t";//单通道数据，每列数据用Tab隔开
				continue;
			}
			for (int channel = 0; channel < cn; channel++)
				file << pData[x + channel] << (channel + 1 < cn ? "," : "\t");//多通道数据，同一列不同通道用','隔开，									//每列数据用Tab隔开
		}
		file << std::endl;
	}
}

void saveAsText(std::string filename, cv::InputArray _src)
{
	std::ofstream outFile(filename.c_str(), std::ios_base::out);
	if (!outFile.is_open())
	{
		std::cout << "Fail to open file (" << filename << ")" << std::endl;
		return;
	}

	cv::Mat src = _src.getMat();//此方法不支持vector<Mat>或vector<vector<>>，具体可参考OpenCV源码(core/src/matrix.cpp)
	CV_Assert(src.empty() == false);

	int depth = src.depth();
	int matStep = (int)(src.step / src.elemSize1());
	const uchar* data = src.data;
	int cn = src.channels();
	int rows = src.rows;
	int cols = src.cols;

	if (depth == CV_8U)
		writeStream(outFile, (const uchar*)data, rows, cols, matStep, cn);
	else if (depth == CV_8S)
		writeStream(outFile, (const schar*)data, rows, cols, matStep, cn);
	else if (depth == CV_16U)
		writeStream(outFile, (const ushort*)data, rows, cols, matStep, cn);
	else if (depth == CV_16S)
		writeStream(outFile, (const short*)data, rows, cols, matStep, cn);
	else if (depth == CV_32S)
		writeStream(outFile, (const int*)data, rows, cols, matStep, cn);
	else if (depth == CV_32F)
	{
		std::streamsize pp = outFile.precision();
		outFile.precision(16);//控制输出精度，可以自行修改
		writeStream(outFile, (const float*)data, rows, cols, matStep, cn);
		outFile.precision(pp);//恢复默认输出精度
	}
	else if (depth == CV_64F)
	{
		std::streamsize pp = outFile.precision();
		outFile.precision(16);
		writeStream(outFile, (const double*)data, rows, cols, matStep, cn);
		outFile.precision(pp);
	}
	else
		std::cout << "unsupported format" << std::endl;
	outFile.close();
}
