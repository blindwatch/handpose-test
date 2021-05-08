#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <io.h>
#include <direct.h>
#include <string>

using namespace std;
using namespace cv;

string save_path = "./calibration_img_camera0/";

int main(int argc, char **argv)
{
	VideoCapture cap;
	cap.open(0);
	if (!cap.isOpened())
	{
		cerr << "Unable to connect to camera" << endl;
		return 1;
	}
	if (argc != 1)
	{
		save_path = argv[1];
	}
	if (_access(save_path.c_str(), 0) == -1)
		_mkdir(save_path.c_str());
	int count = 0;
	Mat frame;
	while (1)
	{
		cap >> frame;
		imshow("picture", frame);
		int key = waitKey(1);
		if (key == 27)
			break;
		else if (key == 115)
		{
			string img_path;
			img_path = save_path + "img_" + to_string(count) + ".jpg";
			imwrite(img_path, frame);
			count++;
		}
	}
	cap.release();
	return 0;
}