#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream>
#include <math.h>
#include <iomanip>

using namespace cv;
using namespace std;

int max_area_col, max_area_row, avegray[4];
//int per_col[1920];
double skewness[3], kurtosis[3], variance[3], temperatureRate;
vector<int> ave_percol(1920), ave_perrow(1080), result_ave10(10), result_averow(5);
//function head
int * calcgray(const Mat& image);
double calcrate(const Mat& image);
int substractionCol(const Mat& image, const Mat& preimage, Mat& drawimage);
int substractionRow(const Mat& image, const Mat& preimage, Mat& drawimage);
double * calcVariance(const Mat& image);
double * calcSkewness(const Mat& image, double * variance);
double * calcKurtosis(const Mat& image, double * variance);
void drawOpticalFlow(const Mat &flowdata, Mat& image, int step);
int  * flameEdge(const Mat & image, Mat & drawimage, vector<Point> &per_col);
void  isolatedFlame(const Mat & image, Mat & drawimage,const int flameline);
double * colAngle(const Mat & drawimage, const vector<Point> result_xy, const int aveline);
vector<vector<int>>  temperature(const Mat & image);

int main(int argc, char** argv) {
	Mat frame, pre_frame, gray_frame, pre_gray_frame, flowdata;
	int imgnum = 4;//image number
	//int shift_distance;
	vector<Point> result_xy;
	vector<vector<int> >Temperature(frame.rows, vector<int>(frame.cols));
	for (int i = 0; i < imgnum; i++) {
	int * flame_line;
	double * angle_point;
		string img_name = "D://ZJU//pictureJiuFeng//坏//" + to_string(i+1) + ".jpg";
		//string img_name = "D:\\ZJU\\pictureJiuFeng\\KNN\\hao\\" + to_string(i + 1) + ".jpg";
		frame = imread(img_name);
		medianBlur(frame, frame, 3);
		frame.copyTo(pre_frame);
		cvtColor(frame, gray_frame, COLOR_BGR2GRAY);
		//Temperature = temperature(frame);
		calcgray(pre_frame);//计算均值
		temperatureRate = calcrate(gray_frame);//计算高温率
		calcVariance(pre_frame);//方差
		calcSkewness(pre_frame,variance);//偏度
		calcKurtosis(pre_frame, variance);//峰度
		flame_line = flameEdge(frame, pre_frame, result_xy);
		isolatedFlame(frame, pre_frame,flame_line[0]);
		angle_point = colAngle(pre_frame, result_xy, flame_line[0]);
		//shift_distance = pow(pow(flame_line[1] - 868, 2) + pow(flame_line[2] - 205, 2), 0.5);
		 if (!pre_gray_frame.empty()) {
			//max_area_col = substractionCol(gray_frame, pre_gray_frame, pre_frame);//帧差法和画图
			//max_area_row = substractionRow(gray_frame, pre_gray_frame, pre_frame);
			//calcOpticalFlowFarneback(pre_gray_frame, gray_frame, flowdata, 0.5, 3, 15, 3, 5, 1.2, 0);
			//drawOpticalFlow(flowdata, pre_frame, 5);
		}
		/*数据保存
		fstream output_stream;
		output_stream.open("D:\\FlameDataDD.txt", ios::out | ios::app);
		output_stream << avegray[3]<<"\t"<< avegray[2] << "\t" << avegray[1] << "\t" << avegray[0] << "\t" << setiosflags(ios::fixed) << setprecision(2) <<
			variance[2] << "\t" << variance[1] << "\t" << variance[0] << "\t" << setiosflags(ios::fixed) << setprecision(2) <<
			skewness[2] << "\t" << skewness[1] << "\t" << skewness[0] << "\t" << setiosflags(ios::fixed) << setprecision(2) <<
			kurtosis[2] << "\t" << kurtosis[1] << "\t" << kurtosis[0] << "\t" << setiosflags(ios::fixed) << setprecision(2) <<
			temperatureRate << "\t" << max_area_col << "\t" << max_area_row << "\t" << setiosflags(ios::fixed) << setprecision(2) <<
			flame_line[1] << "\t" << flame_line[2] << "\t" << flame_line[3] << "\t" << flame_line[4] << "\t" << flame_line[5] << "\t" << flame_line[6] << "\t" << setiosflags(ios::fixed) << setprecision(4)<<
			angle_point[0] << "\t" <<angle_point[1] << "\t" << angle_point[2] <<"\t" << angle_point[3] <<"\t" << angle_point[4] << "\t" << angle_point[5] << endl;
			*/


		//printf("高温率为：%f\%%\t列最大变化区域在：%d\t行最大变化区域在：%d\n", temperatureRate, max_area_col, max_area_row);
		//printf("灰度均值：%d\n", avegray[3]);
		//printf("红色通道：\n均值：%d\t方差为：%f\t偏度为：%f\t峰度为：%f\n", avegray[2], variance[2], skewness[2], kurtosis[2]);
		//printf("绿色通道：\n均值：%d\t方差为：%f\t偏度为：%f\t峰度为：%f\n", avegray[1], variance[1], skewness[1], kurtosis[1]);
		//printf("蓝色通道：\n均值：%d\t方差为：%f\t偏度为：%f\t峰度为：%f\n", avegray[0], variance[0], skewness[0], kurtosis[0]);
		//printf("六个区域纵向偏移角度：\n%f\t%f\t%f\t%f\t%f\t%f\n", angle_point[0], angle_point[1], angle_point[2], angle_point[3], angle_point[4], angle_point[5]);
		cout  << flame_line[0]<<endl;
		/*printf("质心偏移距离：%d\n", shift_distance);
		for (int i = 0; i < 6; i++) {
			cout << "纵向偏移角度:\t" << angle_point[i] << endl;
		}
		 
		 for (int i = 0; i < frame.rows; i++) {
			 for (int k = 0; k < frame.cols; k++) {
				 if (k == 0 && i != 0) {
					 cout << Temperature[i][k] << endl;
				 }
				 else {
					 cout << Temperature[i][k] << "\t";
				 }
			 }
		 }*/

		//delete[] angle_point;
		delete [] flame_line;
		gray_frame.copyTo(pre_gray_frame);
		namedWindow("video", CV_WINDOW_KEEPRATIO);
		imshow("video", pre_frame);
		char c = waitKey(1000);
		if (c == 27) {
		cout << "stop" << endl;
		break;
		}
	}

	waitKey(0);
	return 0;
}




int * calcgray(const Mat& image) {
	int height = 650;
	int width = image.cols;
	int sumgray0 = 0;
	int sumgray1 = 0;
	int sumgray2 = 0;
	int sumgray3 = 0;
	Mat gray;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < image.cols; col++) {
			int f0 = image.at<Vec3b>(row, col)[0];//g
			int f1 = image.at<Vec3b>(row, col)[1];//b
			int f2 = image.at<Vec3b>(row, col)[2];//r
			sumgray0 += f0;
			sumgray1 += f1;
			sumgray2 += f2;
			sumgray3 += gray.at<uchar>(row, col);
		}
	}
	avegray[0] = sumgray0 / (height * width);
	avegray[1] = sumgray1 / (height * width);
	avegray[2] = sumgray2 / (height * width);
	avegray[3] = sumgray3 / (height * width);
	
	return avegray;
}

double calcrate(const Mat& image) {
	double sumf = 0;
	int height = 650;
	int width = image.cols;
	double grayrate;
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < image.cols; col++) {
			int f = image.at<uchar>(row, col);
			if (f > 180) sumf++;
			else continue;
		}
	}
	grayrate = (sumf / (height * width)) * 100;
	return grayrate;

}

double * calcVariance(const Mat& image) {
	double numerator0 = { 0 };
	double numerator1 = { 0 };
	double numerator2 = { 0 };
	int height = 650;
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < image.cols; col++) {
			numerator0 += pow((image.at<Vec3b>(row, col)[0] - avegray[0]), 2);
			numerator1 += pow((image.at<Vec3b>(row, col)[1] - avegray[1]), 2);
			numerator2 += pow((image.at<Vec3b>(row, col)[2] - avegray[2]), 2);
		}
	}
	variance[0] = (numerator0 / (height * image.cols - 1));
	variance[1] = (numerator1 / (height * image.cols - 1));
	variance[2] = (numerator2 / (height * image.cols - 1));
	return variance;
}

int substractionCol(const Mat& image, const Mat& preimage, Mat& drawimage) {
	Mat subdata;
	subdata.create(image.size(), image.type());
	int sum = 0;
	int maxArea = 1;
	int maxChange;
	int result_substraction = 0;
	for (int col = 0; col < image.cols; col++) {
		for (int row = 0; row < image.rows; row++) {
			subdata.at<uchar>(row, col) = abs(image.at<uchar>(row, col) - preimage.at<uchar>(row, col));
			sum += subdata.at<uchar>(row, col);
		}
		ave_percol[col] = sum;
		sum = 0;
	}
	int ii = 0;
	for (int i = 0; i < image.cols; i += 192) {
		for (int j = i; (j - i) < 192; j++) {
			result_substraction += ave_percol[j];
		}
		if (i == 0) maxChange = result_substraction;
		result_ave10[ii] = result_substraction;
		if (result_ave10[ii] > maxChange) {
			maxChange = result_ave10[ii];
			maxArea = ii + 1;
		}
		ii++;
		result_substraction = 0;
	}
	/*int k = 2500;
	for (int i = 0; i < 10; i++) {
		Rect rect = Rect(i * 192, 0, 192, cvRound(result_ave10[i] / k));
		Scalar color = Scalar(255, 0, 0);
		rectangle(drawimage, rect, color, 2, LINE_8);
	}
		Rect rect = Rect((maxArea-1) * 128, 0, 128, 720);
		Scalar color = Scalar(0, 0, 255);
		rectangle(drawimage, rect, color, 2, LINE_8);*/
	
	return maxArea;
}

int substractionRow(const Mat& image, const Mat& preimage, Mat& drawimage) {
	Mat subdata;
	subdata.create(image.size(), image.type());
	int sum = 0;
	int maxArea = 1;
	int maxChange;
	int result_substraction = 0;
	for (int row = 0; row < image.rows; row++) {
		for (int col = 0; col < image.cols; col++) {
			subdata.at<uchar>(row, col) = abs(image.at<uchar>(row, col) - preimage.at<uchar>(row, col));
			sum += subdata.at<uchar>(row, col);
		}
		ave_perrow[row] = sum;
		sum = 0;
	}
	int ii = 0;
	for (int i = 0; i < image.rows; i += 216) {
		for (int j = i; (j - i) < 216; j++) {
			result_substraction += ave_perrow[j];
		}
		if (i == 0) maxChange = result_substraction;
		result_averow[ii] = result_substraction;
		if (result_averow[ii] > maxChange) {
			maxChange = result_averow[ii];
			maxArea = ii + 1;
		}
		ii++;
		result_substraction = 0;
	}

	/*int k = 4000;
	for (int i = 0; i < 5; i++) {
		Rect rect = Rect(0,i * 216, cvRound(result_averow[i] / k),216);
		Scalar color = Scalar(0, 255, 0);
		rectangle(drawimage, rect, color, 2, LINE_8);
	}

	/*Rect rect = Rect(0, (maxArea - 1) * 144, 1280, 144);
	Scalar color = Scalar(0, 0, 255);
	rectangle(drawimage, rect, color, 2, LINE_8);*/

	return maxArea;
}

double * calcSkewness(const Mat& image, double * variance) {
	double numerator0 = 0;
	double numerator1 = 0;
	double numerator2 = 0;
	int height = 650;
	for (int row = 0; row <height; row++) {
		for (int col = 0; col < image.cols; col++) {
			numerator0 += pow((image.at<Vec3b>(row, col)[0] - avegray[0]), 3);
			numerator1 += pow((image.at<Vec3b>(row, col)[0] - avegray[1]), 3);
			numerator2 += pow((image.at<Vec3b>(row, col)[0] - avegray[2]), 3);
		}
	}
	skewness[0] = numerator0 / (height * image.cols * pow(variance[0], 1.5));
	skewness[1] = numerator1 / (height * image.cols * pow(variance[1], 1.5));
	skewness[2] = numerator2 / (height * image.cols * pow(variance[2], 1.5));
	return skewness;
}

double * calcKurtosis(const Mat& image, double * variance) {
	double numerator0 = 0;
	double numerator1 = 0;
	double numerator2 = 0;
	int height = 650;
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < image.cols; col++) {
			numerator0 += pow((image.at<Vec3b>(row, col)[0] - avegray[0]), 4);
			numerator1 += pow((image.at<Vec3b>(row, col)[1] - avegray[1]), 4);
			numerator2 += pow((image.at<Vec3b>(row, col)[2] - avegray[2]), 4);
		}
	}
	kurtosis[0] = numerator0 / (height * image.cols * pow(variance[0], 2));
	kurtosis[1] = numerator0 / (height * image.cols * pow(variance[1], 2));
	kurtosis[2] = numerator0 / (height * image.cols * pow(variance[2], 2));

	return kurtosis;
}

void drawOpticalFlow(const Mat &flowdata, Mat& image, int step) {

	for (int col = 0; col < image.cols; col+=step) {
		for (int row = 0; row < image.rows; row+=step) {
			const Point2f fxy = flowdata.at<Point2f>(row, col);
			if ((fxy.x > 1 || fxy.y > 1) && row > 500) {
				arrowedLine(image, Point(col, row), Point(cvRound(col + 2 * fxy.x), cvRound(row + 2 * fxy.y)), Scalar(0, 255, 0), 1, 8, 0);
			}
		}
	}
}

int * flameEdge(const Mat & image, Mat & drawimage, vector<Point>& per_col) {
	per_col.clear();
	Mat grayframe, threshold_image, xgrad, ygrad;
	int * aveline = new int[7];
	cvtColor(image, grayframe, COLOR_BGR2GRAY);
	int g_th = 170;//高温阈值
	long  long int fenziX;
	int fenmu, CentroidX;
	//long long int fenziY;
	for (int k = 0; k < 6; k++) {
		fenziX = 0;
		fenmu = 0;
		CentroidX = 0;
		for (int col = k * 320; col < (k + 1) * 320; col++) {
			for (int row = 0; row < 650; row++) {
				if (grayframe.at<uchar>(row, col) > g_th) {
					fenziX += grayframe.at<uchar>(row, col) * col;
					//fenziY += gray_frame.at<uchar>(row, col) * row;
					fenmu += grayframe.at<uchar>(row, col);
				}
				else {
					continue;
				}
			}
		}
		if (fenmu != 0) {
			CentroidX = fenziX / fenmu;
		}
		else {
			CentroidX = k * 320 + 159;
		}
		//CentroidY = fenziY / fenmu;
		aveline[k + 1] = CentroidX;
		//cout << CentroidX << "\t" << CentroidY << endl;
		circle(drawimage, Point(k * 320 + 159, 205), 8, Scalar(0, 255, 0), -1);
		circle(drawimage, Point(CentroidX, 205), 8, Scalar(0, 0, 255), -1);
		line(drawimage, Point(k * 320 + 159, 205), Point(CentroidX, 205), Scalar(255, 0, 0), 2, 8, 0);
	}
	threshold(grayframe, threshold_image, 120, 255, CV_THRESH_BINARY);
	//threshold(dst2, dst2, 45, 255, CV_THRESH_BINARY);
	Mat kernel = getStructuringElement(MORPH_RECT, Size(25, 25), Point(-1, -1));
	Mat kerne2 = getStructuringElement(MORPH_RECT, Size(10, 10), Point(-1, -1));
	morphologyEx(threshold_image, threshold_image, CV_MOP_CLOSE, kernel);
	erode(threshold_image, threshold_image, kerne2);
	Scharr(threshold_image, xgrad, CV_16S, 1, 0);
	Scharr(threshold_image, ygrad, CV_16S, 0, 1);
	convertScaleAbs(xgrad, xgrad);
	convertScaleAbs(ygrad, ygrad);
	Mat xygrad = Mat(xgrad.size(), xgrad.type());
	int width = xgrad.cols;
	int height = ygrad.rows;
	for (int col = 0; col < width; col++) {
		for (int row = 0; row < height; row++) {
			if (row > 950) {
				xygrad.at<uchar>(row, col) = 0;
			}
			else {
				int xg = xgrad.at<uchar>(row, col);
				int yg = ygrad.at<uchar>(row, col);
				int xy = xg + yg;
				xygrad.at<uchar>(row, col) = saturate_cast<uchar>(xy);
			}
		}
	}
	vector<vector<Point>> contours;
	vector<Vec4i> hierachy;
	findContours(xygrad, contours, hierachy, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));
	size_t maxnumber = 0;
	size_t maxsize = contours[0].size();
	//cout << contours.size() << endl;
	for (size_t i = 0; i < contours.size(); i++) {
		//cout << contours[i].size() << endl;
		if (contours[i].size() > maxsize) {
			maxsize = contours[i].size();
			maxnumber = i;
		}
	}
	drawContours(drawimage, contours, maxnumber, Scalar(255, 255, 0), 5, 8, hierachy, 0, Point(0, 0));
	long int aveline0 = 0;
	for (size_t ai = 0; ai < contours[maxnumber].size(); ai++) {
		per_col.push_back(Point(contours[maxnumber][ai].x, contours[maxnumber][ai].y));
		aveline0 += contours[maxnumber][ai].y;
	}
	aveline0 = aveline0 / static_cast<int> (contours[maxnumber].size());
	line(drawimage, Point(0, aveline0), Point(1920, aveline0), Scalar(0, 0, 255), 2, 8, 0);
	aveline[0] = aveline0;
	namedWindow("xygrad", CV_WINDOW_KEEPRATIO);
	imshow("xygrad", xygrad);
	return aveline;
}

void  isolatedFlame(const Mat & image, Mat & drawimage,const int flameline) {
	int minrgb = 256;
	int I;
	Mat gray_frame,raw_image;
	raw_image = image(Range(0,image.rows),Range(0,image.cols));
	cvtColor(raw_image, gray_frame, COLOR_BGR2GRAY);
	Mat dst2 = Mat::zeros(gray_frame.size(), gray_frame.type());
	for (int row = 0; row < raw_image.rows; row++) {
		for (int col = 0; col < raw_image.cols; col++) {
			if (raw_image.at<Vec3b>(row, col)[0] <raw_image.at<Vec3b>(row, col)[1]) {
				minrgb = raw_image.at<Vec3b>(row, col)[0];
			}
			else {
				minrgb = raw_image.at<Vec3b>(row, col)[1];
			}
			if (raw_image.at<Vec3b>(row, col)[2] < minrgb) {
				minrgb = raw_image.at<Vec3b>(row, col)[2];
			}

			I = (raw_image.at<Vec3b>(row, col)[0] + raw_image.at<Vec3b>(row, col)[1] + raw_image.at<Vec3b>(row, col)[2]) / 3;
			if (I > 0) {
				dst2.at<uchar>(row, col) = saturate_cast<uchar>(((1 - minrgb / I) * 100) - 55 * (255 - (raw_image.at<Vec3b>(row, col)[2])) / 135);
			}
			minrgb = 256;
		}
	}
	threshold(dst2, dst2, 45, 255, CV_THRESH_BINARY);
	Mat kernel1 = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	Mat kernel2 = getStructuringElement(MORPH_RECT, Size(5, 5), Point(-1, -1));
	morphologyEx(dst2, dst2, CV_MOP_CLOSE, kernel1, Point(-1, -1), 1);
	dilate(dst2, dst2, kernel2, Point(-1, -1), 4);// 膨胀*/
	vector<vector<Point>> contours;
	vector<Vec4i> hireachy;
	Rect roi;
	findContours(dst2, contours, hireachy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));
	if (contours.size() > 0) {
		double minArea = 500.0;
		//double maxArea = 20000.0;
		for (size_t t = 0; t < contours.size(); t++) {
			double area = contourArea(contours[static_cast<int>(t)]);
			if (area > minArea) {
				roi = boundingRect(contours[static_cast<int>(t)]);
				bool a1 = roi.x > 1400 && roi.y > 950;
				bool a2 = roi.x < 890 && roi.y < 125;
				bool a3 = roi.y < flameline;
				if (a1 || a2 || a3) {
					continue;
				}
				else {
					rectangle(drawimage, roi, Scalar(0, 0, 255), 3, 8, 0);
				}
			}
		}
	}
	namedWindow("GULI", CV_WINDOW_KEEPRATIO);
	imshow("GULI", dst2);
}

double * colAngle(const Mat & drawimage, const vector<Point> result_xy, const int aveline) {
	//纵向偏移角度计算(5区域)
	double * angle = new double[6];
	int ii = 0;
	//result_xy.size()=755;
	int maxy, miny;
	double minmax;
	for (int col = 0; col < drawimage.cols; col += 320) {
		maxy = aveline;
		miny = aveline;
		minmax = 0;
		for (int iter = 9; iter < result_xy.size()-10; iter++) {
			if (result_xy[iter].y > maxy && result_xy[iter].x < (ii + 1) * 320 && result_xy[iter].x >= ii * 320) {
				maxy = result_xy[iter].y;
			}
			if (result_xy[iter].y < miny && result_xy[iter].x < (ii + 1) * 320 && result_xy[iter].x >= ii * 320 ) {
				miny = result_xy[iter].y;
			}
		}

		if ((maxy - aveline) >= (aveline - miny)) {
			minmax = maxy - aveline;
		}
		else {
			minmax = miny - aveline;
		}
		
		line(drawimage, Point(ii * 320, aveline), Point((ii + 1) * 320, aveline+minmax), Scalar(255, 0, 255), 2, 8, 0);
		double tan = minmax / 320;
		angle[ii] = atan(tan);
		ii++;
	}
	return angle;
}

vector<vector<int>>  temperature(const Mat & image) {
	int height = image.rows;
	int width = image.cols;
	vector<vector<int> >T(height, vector<int>(width));
	Mat T_map;
	T_map.create(image.size(), image.type());
	double c2 = 1.4388e-2;
	double R_wavelen = 625e-9;
	double G_wavelen = 520e-9;
	double t,I_R, I_G;
	int kr = 1e6;
	int kg = 5e5;

	for (int row = 0; row < image.rows; row++) {
		for (int col = 0; col < image.cols; col++) {
			I_R = image.at<Vec3b>(row, col)[2] * kr;
			I_G = image.at<Vec3b>(row, col)[1] * kg;
			t = -c2 * (1 / R_wavelen - 1 / G_wavelen) / log(I_R * pow(R_wavelen, 5) / I_G / pow(G_wavelen, 5));
			T[row][col] = round(t);
			if (0 < T[row][col] && T[row][col] < 1600) {
				T_map.at<Vec3b>(row, col)[2] = 50;
				T_map.at<Vec3b>(row, col)[1] = 200;
				T_map.at<Vec3b>(row, col)[0] = 50;
			}
			else if (1600 < T[row][col] && T[row][col] < 1800) {
				T_map.at<Vec3b>(row, col)[2] = 200;
				T_map.at<Vec3b>(row, col)[1] = 255;
				T_map.at<Vec3b>(row, col)[0] = 200;
			}
			else if (1800 < T[row][col] && T[row][col] < 1900) {
				T_map.at<Vec3b>(row, col)[2] = 255;
				T_map.at<Vec3b>(row, col)[1] = 255;
				T_map.at<Vec3b>(row, col)[0] = 150;
			}
			else if (1900 < T[row][col] && T[row][col] < 2000) {
				T_map.at<Vec3b>(row, col)[2] = 255;
				T_map.at<Vec3b>(row, col)[1] = 150;
				T_map.at<Vec3b>(row, col)[0] = 100;
			}
			else if (2000 < T[row][col] && T[row][col] < 2200) {
				T_map.at<Vec3b>(row, col)[2] = 255;
				T_map.at<Vec3b>(row, col)[1] = 100;
				T_map.at<Vec3b>(row, col)[0] = 150;
			}
			else  {
				T_map.at<Vec3b>(row, col)[2] = 255;
				T_map.at<Vec3b>(row, col)[1] = 0;
				T_map.at<Vec3b>(row, col)[0] = 0;
			}
		}
	}
	namedWindow("Temperature", CV_WINDOW_KEEPRATIO);
	imshow("Temperature",T_map);
	return T;
}
