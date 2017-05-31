// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <iostream>

using namespace std;

bool inBounds(Mat img, int u, int v) {
	if (u >= 0 && u<img.rows && v >= 0 && v<img.cols)
		return true;
	else return false;
}

void harrisCornerDetection() {
	int corners = 0;
	Mat src, x2y2, xy, trace, src_gray, derivata_X, derivata_Y, x2_deriv, y2_deriv,
		derivata_XY, x2g_deriv, y2g_deriv, xyg_deriv, dst, dst_norm, dst_norm_scaled;

	int thresh = 128;
	src = imread("Images/imaginiPI/2.png", CV_LOAD_IMAGE_GRAYSCALE);

	imshow("imgorig", src);

	src_gray = src.clone();
	//calculam derivatele partiale in x si y pentru matricea initiala
	Sobel(src_gray, derivata_X, CV_32FC1, 1, 0, 3, BORDER_DEFAULT);
	Sobel(src_gray, derivata_Y, CV_32FC1, 0, 1, 3, BORDER_DEFAULT);
	//calculam matricile de care avem nevoie mai departe:
	//Ix^2, Iy^2 si Ix * Iy
	pow(derivata_X, 2.0, x2_deriv);
	pow(derivata_Y, 2.0, y2_deriv);
	multiply(derivata_X, derivata_Y, derivata_XY);
	//derivatele imaginii introduc zgomot, deci va trebui sa-l eliminam
	//din toate matricile calculate pana acum
	GaussianBlur(x2_deriv, x2g_deriv, Size(7, 7), 2.0, 0.0, BORDER_DEFAULT);
	GaussianBlur(y2_deriv, y2g_deriv, Size(7, 7), 0.0, 2.0, BORDER_DEFAULT);
	GaussianBlur(derivata_XY, xyg_deriv, Size(7, 7), 2.0, 2.0, BORDER_DEFAULT);
	//matricea rezultat(M) este de forma:
	//		Ix^2		Ix * Iy
	//		Ix * Iy		Iy^2
	//matricea destinatie cu ajutorul careia determinam colturile in
	//imaginea initiala este data de relatia:
	//R = det(M) - k * (trace(M))^2, unde det = determinantul lui M
	//trace = urma matricii M, k = Harris detector free parameter
	multiply(x2g_deriv, y2g_deriv, x2y2);
	multiply(xyg_deriv, xyg_deriv, xy);
	pow((x2g_deriv + y2g_deriv), 2.0, trace);
	dst = (x2y2 - xy) - 0.04 * trace;
	//normalizam matricea finala pentru a o aduce in domeniul 0-255
	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	//parcurgem matricea initiala si desenam cercuri unde valoarea
	//din matricea finala este mai mare decat un anumit threshold
	for (int i = 0; i < src_gray.rows; i++)
	{
		for (int j = 0; j < src_gray.cols; j++)
		{
			if ((int)dst_norm.at<float>(i, j) > thresh)
			{
				circle(src_gray, Point(j, i), 5, Scalar(0), 2, 8, 0);

				corners++;
			}
		}
	}
	imshow("imgdst", src_gray);
	std::cout << corners;
	waitKey(0);
}


void shiTomasiCornerDetection() {
	Mat src, x2y2, xy, trace, src_clone, xDeriv, yDeriv, x2Deriv, y2Deriv,
		xyDeriv, x2DerivGauss, y2DerivGauss, xyDerivGauss, dst, dst_norm, det, det_norm, trace_norm;
	int thresh = 128;
	int cornersFound = 0;
	int expectedCorners = 200;
	src = imread("Images/imaginiPI/2.png", CV_LOAD_IMAGE_GRAYSCALE);
	dst = Mat(src.rows, src.cols, CV_32FC1);
	imshow("imgorig", src);

	src_clone = src.clone();
	Sobel(src_clone, xDeriv, CV_32FC1, 1, 0, 3, BORDER_DEFAULT);
	Sobel(src_clone, yDeriv, CV_32FC1, 0, 1, 3, BORDER_DEFAULT);
	pow(xDeriv, 2.0, x2Deriv);
	pow(yDeriv, 2.0, y2Deriv);
	multiply(xDeriv, yDeriv, xyDeriv);
	GaussianBlur(x2Deriv, x2DerivGauss, Size(7, 7), 2.0, 0.0, BORDER_DEFAULT);
	GaussianBlur(y2Deriv, y2DerivGauss, Size(7, 7), 0.0, 2.0, BORDER_DEFAULT);
	GaussianBlur(xyDeriv, xyDerivGauss, Size(7, 7), 2.0, 2.0, BORDER_DEFAULT);
	multiply(x2DerivGauss, y2DerivGauss, x2y2);
	multiply(xyDerivGauss, xyDerivGauss, xy);
	//pow((x2DerivGauss + y2DerivGauss), 2.0, trace);
	trace = x2DerivGauss + y2DerivGauss;
	det = x2y2 - xy;

	/*if (x2DerivGauss.data<y2DerivGauss.data) {
	dst = x2DerivGauss;
	}
	else { dst = y2DerivGauss; }*/
	normalize(det, det_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	normalize(trace, trace_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());


	//std::cout << "aici";
	for (int i = 0; i < src_clone.rows; i++)
	{
		for (int j = 0; j < src_clone.cols; j++)
		{
			float a = 1, b = -(trace.at<float>(i, j)), c = det.at<float>(i, j), x1 = 0, x2 = 0, delta = 0;
			delta = b*b - 4 * a*c;
			//	std::cout << "aici \n";
			if (delta > 0) {
				x1 = (-b + sqrt(delta)) / (2 * a);
				x2 = (-b - sqrt(delta)) / (2 * a);
			}

			else if (delta == 0) {
				x1 = (-b + sqrt(delta)) / (2 * a);
				x2 = x1;
			}
			//	std::cout << "aici \n";
			if (x1 < x2) {
				dst.at<float>(i, j) = x1;
				//	std::cout << "aici \n";
			}
			else {
				dst.at<float>(i, j) = x2;
				//	std::cout << "aici \n";
			}
		}
	}

	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	//normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	for (int i = 0; i < src_clone.rows; i++)
	{
		for (int j = 0; j < src_clone.cols; j++)
		{
			if ((int)dst_norm.at<float>(i, j) > thresh)
			{
				circle(src_clone, Point(j, i), 5, Scalar(0), 1, 8, 0);
				cornersFound++;
			}
		}
	}
	imshow("imgdst", src_clone);
	printf("%d\n", cornersFound);
	printf("%d", expectedCorners);
	waitKey(0);
}

bool testPoint(Point2i p, int i, int j) {
	int di[] = { -1,-1,-1,0,0,0,1,1,1 };
	int dj[] = { -1,0,1,-1,0,1,-1,0,1 };
	for (int k = 0; k < 9; k++) {
		if (!((i < di[k] && j == dj[k]) || (i == di[k] && j > dj[k]) || (i > di[k] && j == dj[k]) || (i == di[k] && j > dj[k]))) {
			return false;
		}
	}
	return true;
}

void fastCornerDetection() {

	Mat src = imread("Images/imaginiPI/2.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat dst = src.clone();
	int threshold = 50;
	int darker = 0;
	int brighter = 0;
	int cornerCount = 0;
	int jAux = -4;
	Point2i point = src.at<uchar>(0, 0);

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			darker = 0;
			brighter = 0;
			int u1 = i - 3;
			int v1 = j;
			int u9 = i + 3;
			int v9 = j;
			if (i>point.x + 1 || j> point.y + 1 || j<point.y - 1) {
				if (inBounds(src, u1, v1) && inBounds(src, u9, v9)) {
					if (src.at<uchar>(i, j) + threshold < src.at<uchar>(u1, v1)) {
						brighter++;
					}
					if (src.at<uchar>(i, j) - threshold > src.at<uchar>(u1, v1)) {
						darker++;
					}
					if (src.at<uchar>(i, j) + threshold < src.at<uchar>(u9, v9)) {
						brighter++;
					}
					if (src.at<uchar>(i, j) - threshold > src.at<uchar>(u9, v9)) {
						darker++;
					}
				}
				if (brighter == 2 || darker == 2) {
					int u5 = i;
					int v5 = j + 3;
					int u13 = i;
					int v13 = j - 3;
					if (inBounds(src, u5, v5) && inBounds(src, u13, v13)) {
						if (src.at<uchar>(i, j) + threshold < src.at<uchar>(u5, v5)) {
							brighter++;
						}
						if (src.at<uchar>(i, j) - threshold > src.at<uchar>(u5, v5)) {
							darker++;
						}
						if (src.at<uchar>(i, j) + threshold < src.at<uchar>(u13, v13)) {
							brighter++;
						}
						if (src.at<uchar>(i, j) - threshold > src.at<uchar>(u13, v13)) {
							darker++;
						}
					}

				}
				if (brighter >= 3 || darker >= 3) {
					circle(dst, Point(j, i), 5, Scalar(0), 1, 8, 0);
					cornerCount++;
					point = Point2i(i, j);
				}
			}
		}
	}
	imshow("src", src);
	imshow("dst", dst);
	cout << cornerCount;
	waitKey(0);

}


int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Harris\n");
		printf(" 2 - Shi-Tomasi\n");
		printf(" 3 - Fast\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
		case 1:
			harrisCornerDetection();
			break;
		case 2:
			shiTomasiCornerDetection();
			break;
		case 3:
			fastCornerDetection();
			break;
		}
	}
	while (op!=0);
	return 0;
}