// Test_vs2017.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//
#include "pch.h"
#include<opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/highgui/highgui_c.h>

using namespace std;
using namespace cv;

Mat src, temp, gray_src, dst;
int t1_value = 40;
int max_value = 255;
const char* OUTPUT_TITLE = "OUTPUT";

void Canny_Demo(int, void*);
void Canny_Demo2(int, void*);
int Max_Entropy(cv::Mat& src, cv::Mat& dst, int thresh, int p);
void ConnectFiltering(cv::Mat & image);
void  Two_PassNew(const Mat &bwImg, Mat &labImg);
cv::Scalar GetRandomColor();
void LabelColor(const cv::Mat& labelImg, cv::Mat& colorLabelImg);
void ConnectFiltering_True(cv::Mat &image);

int main(int argc, char** argv)
{
		
	src = imread("D:\\Programming\\Git_Repository\\Test_vs2017\\Test_vs2017\\Test_vs2017\\A4.jpg");
	if (!src.data)
	{
		cout << "can not find .." << endl;
		return -1;
	}
	//namedWindow("原图", 0);
	imshow("原图", src);

	//高斯模糊 在计算灰度图像的X方向梯度图像以及Y方向梯度图像，求混和后振幅图像，更加清晰的边缘

	cvtColor(src, gray_src, COLOR_BGR2GRAY);
	imshow("灰度图", gray_src);
	//imshow("灰度图像", gray_src);
	//namedWindow(OUTPUT_TITLE,0);

	//createTrackbar("Threshold value: ", OUTPUT_TITLE, &t1_value, max_value, Canny_Demo);
	//Canny_Demo(0, 0);

	//createTrackbar("Threshold value: ", OUTPUT_TITLE, &t1_value, max_value, Canny_Demo2);
	//Canny_Demo2(0,0);


	waitKey(0);
	return 0;
}

void Canny_Demo(int, void*)//直方图均衡化+高斯滤波+Canny边缘检测
{
	Mat dst_after_equalizeHist, dst_after_GaussianBlur;
	equalizeHist(gray_src, dst_after_equalizeHist);//直方图均衡化
	//namedWindow("直方图均衡化后", 0);
	//imshow("直方图均衡化后", dst_after_equalizeHist);
	GaussianBlur(dst_after_equalizeHist, dst_after_GaussianBlur, Size(3, 3), 0, 0);//高斯滤波
	//namedWindow("高斯滤波后", 0);
	//imshow("高斯滤波后", dst_after_GaussianBlur);
	//blur(gray_src, gray_src, Size(3, 3), Point(-1, -1), BORDER_DEFAULT);//均值滤波
	Canny(dst_after_GaussianBlur, dst, t1_value, t1_value * 2, 3, false);
	imshow(OUTPUT_TITLE, dst);
	ConnectFiltering(dst);
}

void Canny_Demo2(int, void*)//高斯滤波+最大熵阈值分割+连通域筛选+Canny算子边缘检测
{
	Mat dst_after_equalizeHist, dst_after_GaussianBlur, dst_after_MaxEntropy;
	equalizeHist(gray_src, dst_after_equalizeHist);//直方图均衡化
	namedWindow("直方图均衡化后", 0);
	imshow("直方图均衡化后", dst_after_equalizeHist);
	GaussianBlur(dst_after_equalizeHist, dst_after_GaussianBlur, Size(3, 3), 0, 0);//高斯滤波
	namedWindow("高斯滤波后", 0);
	imshow("高斯滤波后", dst_after_GaussianBlur);
	int thresh = 0;
	thresh = Max_Entropy(dst_after_GaussianBlur, dst_after_MaxEntropy, thresh, 10); //Max_Entropy
	std::cout << "Mythresh=" << thresh << std::endl;
	namedWindow("最大熵阈值分割后", 0);
	imshow("最大熵阈值分割后", dst_after_MaxEntropy);

	//Canny(dst_after_MaxEntropy, dst, t1_value, t1_value * 2, 3, false);
	//imshow(OUTPUT_TITLE, dst);
}


//最大熵阈值分割函数
int Max_Entropy(cv::Mat& src, cv::Mat& dst, int thresh, int p) {
	const int Grayscale = 256;
	int Graynum[Grayscale] = { 0 };
	int r = src.rows;
	int c = src.cols;
	for (int i = 0; i < r; ++i) {
		const uchar* ptr = src.ptr<uchar>(i);
		for (int j = 0; j < c; ++j) {
			if (ptr[j] == 0)				//排除掉黑色的像素点
				continue;
			Graynum[ptr[j]]++;
		}
	}

	float probability = 0.0; //概率
	float max_Entropy = 0.0; //最大熵
	int totalpix = r * c;
	for (int i = 0; i < Grayscale; ++i) {

		float HO = 0.0; //前景熵
		float HB = 0.0; //背景熵

		//计算前景像素数
		int frontpix = 0;
		for (int j = 0; j < i; ++j) {
			frontpix += Graynum[j];
		}
		//计算前景熵
		for (int j = 0; j < i; ++j) {
			if (Graynum[j] != 0) {
				probability = (float)Graynum[j] / frontpix;
				HO = HO + probability * log(1 / probability);
			}
		}

		//计算背景熵
		for (int k = i; k < Grayscale; ++k) {
			if (Graynum[k] != 0) {
				probability = (float)Graynum[k] / (totalpix - frontpix);
				HB = HB + probability * log(1 / probability);
			}
		}

		//计算最大熵
		if (HO + HB > max_Entropy) {
			max_Entropy = HO + HB;
			thresh = i + p;
		}
	}

	//阈值处理
	src.copyTo(dst);
	for (int i = 0; i < r; ++i) {
		uchar* ptr = dst.ptr<uchar>(i);
		for (int j = 0; j < c; ++j) {
			if (ptr[j] > thresh)
				ptr[j] = 255;
			else
				ptr[j] = 0;
		}
	}
	return thresh;
}

//连通的轮廓筛选
void ConnectFiltering(cv::Mat & image) 
{
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(image, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point());
	Mat imageContours = Mat::zeros(image.size(), CV_8UC1);
	Mat Contours = Mat::zeros(image.size(), CV_8UC1);  //绘制
	double minarea = 100;
	int k = 0;
	double tmparea = 0.0;
	for (int i = 0; i < contours.size(); i++)
	{
		//contours[i]代表的是第i个轮廓，contours[i].size()代表的是第i个轮廓上所有的像素点数
		if (contours[i].size() > minarea)
		{
			for (int j = 0; j < contours[i].size(); j++)
			{
				//绘制出contours向量内所有的像素点
				Point P = Point(contours[i][j].x, contours[i][j].y);
				Contours.at<uchar>(P) = 255;
			}

			//输出hierarchy向量内容
			char ch[256];
			sprintf_s(ch, "%d", i);
			string str = ch;
			cout << "第" << k << "个有效连通域" << endl;
			k++;
			cout << "向量hierarchy的第" << str << " 个元素内容为：" << endl << hierarchy[i] << endl << endl;
			//绘制轮廓
			drawContours(imageContours, contours, i, Scalar(255), 1, 8, hierarchy);
		}
	}
	namedWindow("所有连通域", 0);
	imshow("所有连通域", imageContours); //轮廓
	//imshow("Point of Contours", Contours);   //向量contours内保存的所有轮廓点集
}



//------------------------------【两步法新改进版】----------------------------------------------
// 对二值图像进行连通区域标记,从1开始标号
void  Two_PassNew(const Mat &bwImg, Mat &labImg)
{
	assert(bwImg.type() == CV_8UC1);
	labImg.create(bwImg.size(), CV_32SC1);   //bwImg.convertTo( labImg, CV_32SC1 );
	labImg = Scalar(0);
	labImg.setTo(Scalar(1), bwImg);
	assert(labImg.isContinuous());
	int label = 1;
	const int Rows = bwImg.rows - 1, Cols = bwImg.cols - 1;
	vector<int> labelSet;
	labelSet.push_back(0);
	labelSet.push_back(1);
	//the first pass
	int *data_prev = (int*)labImg.data;   //0-th row : int* data_prev = labImg.ptr<int>(i-1);
	int *data_cur = (int*)(labImg.data + labImg.step); //1-st row : int* data_cur = labImg.ptr<int>(i);
	for (int i = 1; i < Rows; i++)
	{
		data_cur++;
		data_prev++;
		for (int j = 1; j < Cols; j++, data_cur++, data_prev++)
		{
			if (*data_cur != 1)
				continue;
			int left = *(data_cur - 1);
			int up = *data_prev;
			int neighborLabels[2];
			int cnt = 0;
			if (left > 1)
				neighborLabels[cnt++] = left;
			if (up > 1)
				neighborLabels[cnt++] = up;
			if (!cnt)
			{
				labelSet.push_back(++label);
				labelSet[label] = label;
				*data_cur = label;
				continue;
			}
			int smallestLabel = neighborLabels[0];
			if (cnt == 2 && neighborLabels[1] < smallestLabel)
				smallestLabel = neighborLabels[1];
			*data_cur = smallestLabel;
			// 保存最小等价表
			for (int k = 0; k < cnt; k++)
			{
				int tempLabel = neighborLabels[k];
				int& oldSmallestLabel = labelSet[tempLabel];  //这里的&不是取地址符号,而是引用符号
				if (oldSmallestLabel > smallestLabel)
				{
					labelSet[oldSmallestLabel] = smallestLabel;
					oldSmallestLabel = smallestLabel;
				}
				else if (oldSmallestLabel < smallestLabel)
					labelSet[smallestLabel] = oldSmallestLabel;
			}
		}
		data_cur++;
		data_prev++;
	}
	//更新等价队列表,将最小标号给重复区域
	for (size_t i = 2; i < labelSet.size(); i++)
	{
		int curLabel = labelSet[i];
		int prelabel = labelSet[curLabel];
		while (prelabel != curLabel)
		{
			curLabel = prelabel;
			prelabel = labelSet[prelabel];
		}
		labelSet[i] = curLabel;
	}
	//second pass
	data_cur = (int*)labImg.data;
	for (int i = 0; i < Rows; i++)
	{
		for (int j = 0; j < bwImg.cols - 1; j++, data_cur++)
			*data_cur = labelSet[*data_cur];
		data_cur++;
	}
}



//---------------------------------【颜色标记程序】-----------------------------------
//彩色显示
cv::Scalar GetRandomColor()
{
	uchar r = 255 * (rand() / (1.0 + RAND_MAX));
	uchar g = 255 * (rand() / (1.0 + RAND_MAX));
	uchar b = 255 * (rand() / (1.0 + RAND_MAX));
	return cv::Scalar(b, g, r);
}


void LabelColor(const cv::Mat& labelImg, cv::Mat& colorLabelImg)
{
	int num = 0;
	if (labelImg.empty() ||
		labelImg.type() != CV_32SC1)
	{
		return;
	}

	std::map<int, cv::Scalar> colors;

	int rows = labelImg.rows;
	int cols = labelImg.cols;

	colorLabelImg.release();
	colorLabelImg.create(rows, cols, CV_8UC3);
	colorLabelImg = cv::Scalar::all(0);

	for (int i = 0; i < rows; i++)
	{
		const int* data_src = (int*)labelImg.ptr<int>(i);
		uchar* data_dst = colorLabelImg.ptr<uchar>(i);
		for (int j = 0; j < cols; j++)
		{
			int pixelValue = data_src[j];
			if (pixelValue > 1)
			{
				if (colors.count(pixelValue) <= 0)
				{
					colors[pixelValue] = GetRandomColor();
					num++;
				}

				cv::Scalar color = colors[pixelValue];
				*data_dst++ = color[0];
				*data_dst++ = color[1];
				*data_dst++ = color[2];
			}
			else
			{
				data_dst++;
				data_dst++;
				data_dst++;
			}
		}
	}
	printf("color num : %d \n", num);
}

//使用连通域筛选
void ConnectFiltering_True(cv::Mat &image)
{
	Mat dst_after_equalizeHist, dst_after_GaussianBlur, dst_after_MaxEntropy;
	//equalizeHist(gray_src, dst_after_equalizeHist);//直方图均衡化
	////namedWindow("直方图均衡化后", 0);
	//imshow("直方图均衡化后", dst_after_equalizeHist);
	//GaussianBlur(gray_src, dst_after_GaussianBlur, Size(3, 3), 0, 0);//高斯滤波
	////namedWindow("高斯滤波后", 0);
	//imshow("高斯滤波后", dst_after_GaussianBlur);
	//二值化
	threshold(image, dst_after_MaxEntropy, 170, 255, CV_THRESH_BINARY);
	//int thresh = 0;
	//thresh = Max_Entropy(gray_src, dst_after_MaxEntropy, thresh, 10); //Max_Entropy
	//std::cout << "Mythresh=" << thresh << std::endl;
	//namedWindow("最大熵阈值分割后", 0);
	imshow("最大熵阈值分割后", dst_after_MaxEntropy);
	Mat dst_after_dilate, dst_after_erode;
	//1.开运算
	//腐蚀
	//Mat element = getStructuringElement(MORPH_RECT, Size(8, 8));
	//Mat element2 = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	//erode(dst_after_MaxEntropy, dst_after_erode, Mat());
	//膨胀
	dilate(dst_after_MaxEntropy, dst_after_dilate, Mat());
	imshow("开运算后", dst_after_dilate);
	////2.闭运算
	////膨胀
	//dilate(dst_after_dilate, dst_after_dilate, Mat());
	////腐蚀
	//erode(dst_after_dilate, dst_after_erode, Mat());
	//imshow("闭运算后", dst_after_erode);

	cv::Mat labelImg;
	double time;
	time = getTickCount();
	//连通域查找
	Two_PassNew(dst_after_dilate, labelImg);
	time = 1000 * ((double)getTickCount() - time) / getTickFrequency();
	cout << std::fixed << time << "ms" << endl;
	//namedWindow("连通域", 0);
	//imshow("连通域", labelImg);
	//彩色显示
	cv::Mat colorLabelImg;
	LabelColor(labelImg, colorLabelImg);
	cv::imshow("colorImg", colorLabelImg);
	//灰色显示
	cv::Mat grayImg;
	labelImg *= 10;
	labelImg.convertTo(grayImg, CV_8UC1);
	//namedWindow("连通域", 0);
	cv::imshow("连通域", grayImg);
}


