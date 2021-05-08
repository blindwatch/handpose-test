#ifndef __DRAWMODE_H__
#define __DRAWMODE_H__
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/core/cuda.hpp>
#include <math.h>
void drawLine(cv::Mat &mask, cv::Point &pre, cv::Point &now);

void eraseLine(cv::Mat &mask, std::vector<cv::Point> &points);

void display(cv::Mat &frame, cv::Mat &mask);

void clearmask(cv::Mat &mask);
 
void drawsword(cv::Mat &mask, std::vector<cv::Point> &proj);
#endif // !1
