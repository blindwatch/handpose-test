#include "poseRecog.hpp"

int posejudge(const std::vector<cv::Point> &points)
{
	const int POSE_PAIRS[20][2] =
	{
		{0,1}, {1,2}, {2,3}, {3,4},         // thumb
		{0,5}, {5,6}, {6,7}, {7,8},         // index
		{0,9}, {9,10}, {10,11}, {11,12},    // middle
		{0,13}, {13,14}, {14,15}, {15,16},  // ring
		{0,17}, {17,18}, {18,19}, {19,20}   // small
	};
	int state[5] = { 0 };
	std::vector<cv::Point> lines;
	double angle[15] = { 0 };
	for (int i = 0; i < 5; ++i)
	{
		for (int j = 0; j < 4; ++j)
		{
			cv::Point line = cv::Point(points[POSE_PAIRS[i * 4 + j][1]].x - points[POSE_PAIRS[i * 4 + j][0]].x,
				points[POSE_PAIRS[i * 4 + j][1]].y - points[POSE_PAIRS[i * 4 + j][0]].y);
			if (j > 0)
			{
				angle[3*i+j-1] = acos(lines.back().dot(line) / (cv::norm(line) * cv::norm(lines.back()))) * 180 / CV_PI;
			}
			lines.push_back(line);
		}
	}
	for (int i = 0; i < 5; ++i)
	{
		//std::cout << "finger" << i << "angle" << angle[3 * i] << "," << angle[3 * i + 1]
		//	<< "," << angle[3 * i + 2] << std::endl;
		if (angle[3 * i] + angle[3 * i + 1] + angle[3 * i + 2] > 120)
			state[i] = 1;
	}
	if (state[1] == 1)
	{
		return 0;	//抓取物体的状态
	}
	else if (state[2] == 1)
	{
		return 1; //画画模式的绘画状态
	}
	else if (state[3] == 1)
	{
		return 2; //切换到画画状态
	}
	else if (state[4] == 1)
	{
		return 3; //切换到抓取状态
	}
	else
	{
		return 4; //松开或者擦除
	}
}

