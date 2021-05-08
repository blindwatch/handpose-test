#include "drawmode.hpp"

void drawLine(cv::Mat &mask, cv::Point &pre, cv::Point &now)
{
	cv::line(mask, pre, now, cv::Scalar(255, 255, 255), 4);
	pre = now;
	return;
}

void eraseLine(cv::Mat &mask, std::vector<cv::Point> &points)
{
	cv::Point top(points[8]);
	cv::Point bottom(points[17]);
	cv::rectangle(mask, top, bottom, cv::Scalar(0, 0, 0), -1);
	return;
	
}

void display(cv::Mat &frame, cv::Mat &mask)
{
	for (size_t nrow = 0; nrow < frame.rows; ++nrow)
	{
		uchar* mdata = mask.ptr<uchar>(nrow);
		uchar* fdata = frame.ptr<uchar>(nrow);
		for (size_t ncol = 0; ncol < frame.cols * frame.channels(); ncol++)
		{
			if (mdata[ncol] != 0)
			{
				fdata[ncol] = mdata[ncol];
			}
		}
	}

}

void clearmask(cv::Mat &mask)
{
	for (size_t nrow = 0; nrow < mask.rows; ++nrow)
	{
		uchar* mdata = mask.ptr<uchar>(nrow);
		for (size_t ncol = 0; ncol < mask.cols * mask.channels(); ncol++)
		{
			mdata[ncol] = 0;
		}
	}
}

void drawsword(cv::Mat &mask, std::vector<cv::Point> &proj)
{
	cv::line(mask, proj[0], proj[1], cv::Scalar(175, 96, 26), 4);
	cv::line(mask, proj[0], proj[2], cv::Scalar(175, 96, 26), 4);
	cv::line(mask, proj[1], proj[3], cv::Scalar(175, 96, 26), 4);
	cv::line(mask, proj[4], proj[5], cv::Scalar(230, 126, 34), 4);
	cv::line(mask, proj[4], proj[6], cv::Scalar(230, 126, 34), 4);
	cv::line(mask, proj[5], proj[7], cv::Scalar(230, 126, 34), 4);
	cv::line(mask, proj[6], proj[7], cv::Scalar(230, 126, 34), 4);
	cv::line(mask, proj[8], proj[11], cv::Scalar(208, 211, 212), 4);
	cv::line(mask, proj[9], proj[14], cv::Scalar(208, 211, 212), 4);
	cv::line(mask, proj[10], proj[13], cv::Scalar(208, 211, 212), 4);
	cv::line(mask, proj[11], proj[14], cv::Scalar(208, 211, 212), 4);
	cv::line(mask, proj[13], proj[14], cv::Scalar(208, 211, 212), 4);


}