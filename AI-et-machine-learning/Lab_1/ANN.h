#pragma once
#include <vector>
#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/imgproc.hpp>
#include <boost\filesystem.hpp>
#include <boost/algorithm/string.hpp>

namespace MachineLearning
{
	typedef std::vector<std::string>::const_iterator vec_iter;
	class MachineLearning
	{
	public:

		
		MachineLearning()
		{

		}

		struct ImageData
		{
			std::string classname;
			cv::Mat bowFeatures;
		};

		std::string imagesDir;

		int networkInputSize = 768;
		float trainSplitRatio = 0.1f;


		cv::Mat descriptorsSet;
		cv::Mat labels;
		cv::Mat vocabulary;
		cv::Mat testSamples;
		std::vector<std::string> files;
		std::vector<int> testOutputExpected;
		std::vector<MachineLearning::ImageData*> descriptorsMetadata;
		std::set<std::string> classes;
		cv::FlannBasedMatcher flann;
		std::vector<std::vector<int> > confusionMatrix;
		cv::Ptr<cv::ml::ANN_MLP> mlp;




		void train(std::string imgDir);
		void train();


		void testSet(std::string imgDir);
		void testSet();

		std::vector<std::string> getFilesInDirectory(const std::string& directory);

		inline std::string getClassName(const std::string& filename);

		cv::Mat getDescriptors(const cv::Mat& img);

		void readImages(int fileCount, vec_iter begin, vec_iter end, std::function<void(const std::string&, const cv::Mat&)> callback);

		int getClassId(const std::set<std::string>& classes, const std::string& classname);

		cv::Mat getClassCode(const std::set<std::string>& classes, const std::string& classname);

		cv::Mat getBOWFeatures(cv::FlannBasedMatcher& flann, const cv::Mat& descriptors,
			int vocabularySize);

		cv::Ptr<cv::ml::ANN_MLP> getTrainedNeuralNetwork(const cv::Mat& trainSamples,
			const cv::Mat& trainResponses);

		int getPredictedClass(const cv::Mat& predictions);

		std::vector<std::vector<int> > getConfusionMatrix();

		void printConfusionMatrix();

		float getAccuracy();

		void saveModels();

		void loadModel();

		int testThis(int argc, char** argv);

	};
}