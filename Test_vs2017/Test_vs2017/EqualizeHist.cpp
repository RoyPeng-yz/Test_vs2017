// Test_vs2017.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//
#include "pch.h"
#include<opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/highgui/highgui_c.h>

//using namespace std;
//using namespace cv;
//
//class Histogram1D {
//
//private:
//
//	int histSize[1];
//	float hranges[2];
//	const float* ranges[1];
//	int channels[1];
//
//public:
//
//	Histogram1D() {
//		histSize[0] = 256;
//		hranges[0] = 0.0;
//		hranges[1] = 256.0;
//		ranges[0] = hranges;
//		channels[0] = 0;
//	}
//
//	Mat getHistogram(const Mat &image) {
//		Mat hist;
//		calcHist(&image, 1, channels, Mat(), hist, 1, histSize, ranges);
//		return hist;
//	}
//
//	Mat getHistogramImage(const Mat &image, int zoom = 1) {
//		Mat hist = getHistogram(image);
//		return getImageOfHistogram(hist, zoom);
//	}
//
//	static Mat getImageOfHistogram(const Mat &hist, int zoom) {
//		double maxVal = 0;
//		double minVal = 0;
//		minMaxLoc(hist, &minVal, &maxVal, 0, 0);
//
//		int histSize = hist.rows;
//		Mat histImg(histSize*zoom, histSize*zoom, CV_8U, Scalar(255));
//		int hpt = static_cast<int>(0.9*histSize);
//
//		for (int h = 0; h < histSize; h++) {
//			float binVal = hist.at<float>(h);
//			if (binVal > 0) {
//				int intensity = static_cast<int>(binVal*hpt / maxVal);
//				line(histImg, Point(h*zoom, histSize*zoom),
//					Point(h*zoom, (histSize - intensity)*zoom), Scalar(0), zoom);
//			}
//		}
//		return histImg;
//	}
//};
//
//
//
//
//int main()
//{
//	Mat image = imread("A3.bmp", 0);
//	Histogram1D h;
//	Mat histo = h.getHistogram(image);
//	for (int i = 0; i < 256; i++)
//	{
//		if (histo.at<float>(i) != 0) 
//			cout << "Value " << i << " = " << histo.at<float>(i) << endl;
//	}
//	namedWindow("Histogram");
//	imshow("Histogram", h.getHistogramImage(image));
//	namedWindow("Cell",0);
//	imshow("Cell", image);
//	equalizeHist(image, image);
//	namedWindow("CellequalizeHist",0);
//	imshow("CellequalizeHist", image);
//	namedWindow("Histogram2");
//	imshow("Histogram2", h.getHistogramImage(image));
//	waitKey(0);
//	return 0;
//}
