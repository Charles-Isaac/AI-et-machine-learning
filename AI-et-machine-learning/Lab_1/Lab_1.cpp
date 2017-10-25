// Lab 2 - Vision
// Charles-Isaac Côté & Samuel Goulet
// 2017-09-08 ish

#include "stdafx.h"

// Cuda
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// OpenCV
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\types_c.h>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\ml\ml.hpp>
#include "ANN.h" 
using namespace cv;

#include <iostream>

int main()
{
	MachineLearning::MachineLearning ml;
	//Mat m;
	

	std::string imagesDir = "Z:\\Images\\Train\\TrainAnimals";
	ml.trainSplitRatio = 0.001f;
	ml.imagesDir = imagesDir;
	//ml.train(imagesDir);
	ml.loadModel();
	ml.trainSplitRatio = 0.99f;
	ml.testSet();

	// Get confusion matrix of the test set
	ml.getConfusionMatrix();

	// Get accuracy of our model
	std::cout << "Confusion matrix: " << std::endl;
	ml.printConfusionMatrix();
	std::cout << "Accuracy: " << ml.getAccuracy() << std::endl;

	// Save models
	//std::cout << "Saving models..." << std::endl;
	//ml.saveModels();

	/*
	VideoCapture cam;
	if (!cam.open(1)) {
		std::cout << "Can't open the camera!!!" << std::endl;
		waitKey();
		return 1;
	}
	Mat m;
	Mat grayscale;
	Mat bin;
	std::vector<Vec3f> outCircles;
	std::vector<Vec4i> outLines;
	std::vector<Vec2i> inLines;
	std::vector<Vec2i> outDP;
	std::vector<std::vector<Point>> outArrays;
	std::vector<std::vector<Point>> contours;
	std::vector<Vec4i> hierarchy;
	int param1 = 70;
	int param2 = 100;
	double dp = 20.0;
	while (true) {
		cam.read(m);
		Mat OriginaL = m.clone();
		GaussianBlur(m, m, Size(3, 3), 1, 1);
		cvtColor(m, m, COLOR_BGR2HSV);
		Mat hsvmat = m.clone();
		for (size_t j = 0; j < (m.cols * m.rows); j++)
		{
			// Green
			if (m.data[j * 3] >= 60 && m.data[j * 3] <= 80 && m.data[j * 3 + 2] >= 56 && m.data[j * 3 + 2] <= 250 && m.data[j * 3 + 1] > 16) {
				/*m.data[j * 3] = 0;
				m.data[j * 3 + 1] = 0;
				m.data[j * 3 + 2] = 255;*//*
				m.data[j * 3] = 18;
				m.data[j * 3 + 1] = 86;
				m.data[j * 3 + 2] = 55;
			}
			// Red
			else if (m.data[j * 3] >= 0 && m.data[j * 3] <= 5 && m.data[j * 3 + 2] >= 85 && m.data[j * 3 + 2] <= 255 && m.data[j * 3 + 1] > 16) {
				/*m.data[j * 3] = 240;
				m.data[j * 3 + 1] = 255;
				m.data[j * 3 + 2] = 255;*//*
				m.data[j * 3] = 18;
				m.data[j * 3 + 1] = 86;
				m.data[j * 3 + 2] = 55;
			}
			// Yellow
			else if (m.data[j * 3] >= 22 && m.data[j * 3] <= 35 && m.data[j * 3 + 2] >= 170 && m.data[j * 3 + 2] <= 255 && m.data[j * 3 + 1] > 16) {
				m.data[j * 3] = 18;
				m.data[j * 3 + 1] = 86;
				m.data[j * 3 + 2] = 55;
			}
			// Other
			else {
				m.data[j * 3] = 0;
				m.data[j * 3 + 1] = 0;
				m.data[j * 3 + 2] = 0;
			}
		}


		

		int rad = 2;
		Mat phil = Mat(rad, rad, CV_8UC1);
		for (int i = 0; i < rad*rad; i++)
		{
			int x = i % rad;
			int y = i / rad;
			phil.data[i] = (sqrt(x*x + y*y) > rad);
		}






		morphologyEx(m, m, MORPH_OPEN, Mat::ones(Size(6, 6), CV_8UC1));


		morphologyEx(m, m, MORPH_CLOSE, phil, Point(-1, -1), 1, 0, Scalar(0, 0, 0));



		//m = imread("visage.png");

		//GaussianBlur(m, m, Size(9, 9), 2, 2);
		cvtColor(m, grayscale, CV_BGR2GRAY);

		Canny(grayscale, bin, 50, 200, 3);



		imshow("gray", grayscale);
		threshold(bin, bin, 60, 255, THRESH_BINARY);

		findContours(bin, outArrays, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_TC89_KCOS);
		contours.resize(outArrays.size());
		for (int k = 0; k < outArrays.size(); k++) {
			Rect r = boundingRect(outArrays[k]);
			int min = r.width < r.height ? r.width : r.height;
			approxPolyDP(Mat(outArrays[k]), contours[k], min / 7, true);
		}


		unsigned char circles = 0, squares = 0, triangles = 0, rectangles = 0;
		if (contours.size() > 0)
		{
			try {
				int idx = 0;
				for (; idx >= 0; idx = hierarchy[idx][0]) {
					drawContours(m, contours, idx, Scalar(0, 255, 255), 1, 8, hierarchy);
					if (contours[idx].size() == 4) {
						Rect r = boundingRect(contours[idx]);
						float aspectRatio = (float)r.width / r.height;
						if (abs(aspectRatio - 1) < 0.25) {
							squares++;
						}
						else {
							rectangles++;
						}
					}
					else if (contours[idx].size() == 3) {
						triangles++;
					}
					else {
						circles++;
						Rect br = boundingRect(contours[idx]);
						rectangle(m, br, Scalar(255, 0, 0), 1, 8, 0);
					}
				}
			}
			catch (cv::Exception ex) {
				std::cout << ex.msg << std::endl;
			}
		}

		

		std::string text[4] = {
			"Circles " + std::to_string(circles),
			"Squares " + std::to_string(squares),
			"Triangles " + std::to_string(triangles),
			"Rectangles " + std::to_string(rectangles)
		};
		int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
		double fontScale = 1;
		int thickness = 2;
		for (int i = 0; i < 4; i++) {
			cv::Point textOrg(10, i * 28 + 28);
			cv::putText(m, text[i], textOrg, fontFace, fontScale, Scalar(0, 0, 255), thickness, 8);
		}*/

	//	imshow("Nom quelconque", m);
	//	waitKey(1);
	//}

	return 0;
}


