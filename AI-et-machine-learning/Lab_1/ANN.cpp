//https://picoledelimao.github.io/blog/2016/01/31/is-it-a-cat-or-dog-a-neural-network-application-in-opencv/


#include "ANN.h"

namespace MachineLearning
{
	namespace fs = boost::filesystem;

	
	
	
		

		/*
		* Get all files in directory (not recursive)
		* @param directory Directory where the files are contained
		* @return A list containing the file name of all files inside given directory
		**/
		std::vector<std::string> MachineLearning::getFilesInDirectory(const std::string& directory)
		{
			std::vector<std::string> files;
			fs::path root(directory);
			fs::directory_iterator it_end;
			for (fs::directory_iterator it(root); it != it_end; ++it)
			{
				std::string a= it->path().string();
				if (fs::is_regular_file(it->path()))
				{
					files.push_back(it->path().string());
				}
				else
				{
					std::vector<std::string> temp = getFilesInDirectory(it->path().string());
					files.insert(files.end(), temp.begin(), temp.end());
				}
			}
			return files;
		}

		/**
		* Extract the class name from a file name
		*/
		inline std::string MachineLearning::getClassName(const std::string& filename)
		{
			std::string temp = filename.substr(filename.find_last_of('\\') + 1, 1);
			boost::algorithm::to_lower(temp);
			return temp;
		}

		/**
		* Extract local features for an image
		*/
		cv::Mat MachineLearning::getDescriptors(const cv::Mat& img)
		{
			cv::Ptr<cv::KAZE> kaze = cv::KAZE::create();
			std::vector<cv::KeyPoint> keypoints;
			cv::Mat descriptors;
			kaze->detectAndCompute(img, cv::noArray(), keypoints, descriptors);
			return descriptors;
		}

		/**
		* Read images from a list of file names and returns, for each read image,
		* its class name and its local descriptors
		*/
		void MachineLearning::readImages(int fileCount, vec_iter begin, vec_iter end, std::function<void(const std::string&, const cv::Mat&)> callback)
		{
			int idx = 0;
			for (auto it = begin; it != end; ++it)
			{
				idx++;

				//if (idx*1000 % (fileCount) == 0)
				{
					std::cout << "WIP: " << idx*1000 / fileCount / 10.f << "% ..." << std::endl;
				}

				std::string filename = *it;
				//std::cout << "Reading image " << filename << "..." << std::endl;
				cv::Mat img = cv::imread(filename, 0);
				cv::Size size(300, 300);
				cv::resize(img, img, size);
				if (img.empty())
				{
					std::cerr << "WARNING: Could not read image." << std::endl;
					continue;
				}
				std::string classname = getClassName(filename);
				cv::Mat descriptors = getDescriptors(img);
				callback(classname, descriptors);

			}
		}

		/**
		* Transform a class name into an id
		*/
		int MachineLearning::getClassId(const std::set<std::string>& classes, const std::string& classname)
		{
			int index = 0;
			for (auto it = classes.begin(); it != classes.end(); ++it)
			{
				if (*it == classname) break;
				++index;
			}
			return index;
		}

		/**
		* Get a binary code associated to a class
		*/
		cv::Mat MachineLearning::getClassCode(const std::set<std::string>& classes, const std::string& classname)
		{
			cv::Mat code = cv::Mat::zeros(cv::Size((int)classes.size(), 1), CV_32F);
			int index = getClassId(classes, classname);
			code.at<float>(index) = 1;
			return code;
		}

		/**
		* Turn local features into a single bag of words histogram of
		* of visual words (a.k.a., bag of words features)
		*/
		cv::Mat MachineLearning::getBOWFeatures(cv::FlannBasedMatcher& flann, const cv::Mat& descriptors,
			int vocabularySize)
		{
			cv::Mat outputArray = cv::Mat::zeros(cv::Size(vocabularySize, 1), CV_32F);
			std::vector<cv::DMatch> matches;
			flann.match(descriptors, matches);
			for (size_t j = 0; j < matches.size(); j++)
			{
				int visualWord = matches[j].trainIdx;
				outputArray.at<float>(visualWord)++;
			}
			return outputArray;
		}

		/**
		* Get a trained neural network according to some inputs and outputs
		*/
		cv::Ptr<cv::ml::ANN_MLP> MachineLearning::getTrainedNeuralNetwork(const cv::Mat& trainSamples,
			const cv::Mat& trainResponses)
		{
			int networkInputSize = trainSamples.cols;
			int networkOutputSize = trainResponses.cols;
			cv::Ptr<cv::ml::ANN_MLP> mlp = cv::ml::ANN_MLP::create();
			std::vector<int> layerSizes = { networkInputSize, networkInputSize / 2,
				networkOutputSize };
			mlp->setLayerSizes(layerSizes);
			mlp->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM);
			mlp->train(trainSamples, cv::ml::ROW_SAMPLE, trainResponses);
			return mlp;
		}

		/**
		* Receives a column matrix contained the probabilities associated to
		* each class and returns the id of column which contains the highest
		* probability
		*/
		int MachineLearning::getPredictedClass(const cv::Mat& predictions)
		{
			float maxPrediction = predictions.at<float>(0);
			float maxPredictionIndex = 0;
			const float* ptrPredictions = predictions.ptr<float>(0);
			for (int i = 0; i < predictions.cols; i++)
			{
				float prediction = *ptrPredictions++;
				if (prediction > maxPrediction)
				{
					maxPrediction = prediction;
					maxPredictionIndex = i;
				}
			}
			return maxPredictionIndex;
		}

		/**
		* Get a confusion matrix from a set of test samples and their expected
		* outputs
		*/
		std::vector<std::vector<int> > MachineLearning::getConfusionMatrix()
		{
			cv::Mat testOutput;
			int classCount = classes.size();
			mlp->predict(testSamples, testOutput);
			confusionMatrix = std::vector<std::vector<int> >(classCount, std::vector<int>(classCount));
			for (int i = 0; i < testOutput.rows; i++)
			{
				int predictedClass = getPredictedClass(testOutput.row(i));
				int expectedClass = testOutputExpected.at(i);
				confusionMatrix[expectedClass][predictedClass]++;
			}
			return confusionMatrix;
		}

		/**
		* Print a confusion matrix on screen
		*/
		void MachineLearning::printConfusionMatrix()
		{
			for (auto it = classes.begin(); it != classes.end(); ++it)
			{
				std::cout << *it << " ";
			}
			std::cout << std::endl;
			for (size_t i = 0; i < confusionMatrix.size(); i++)
			{
				for (size_t j = 0; j < confusionMatrix[i].size(); j++)
				{
					std::cout << confusionMatrix[i][j] << " ";
				}
				std::cout << std::endl;
			}
		}

		/**
		* Get the accuracy for a model (i.e., percentage of correctly predicted
		* test samples)
		*/
		float MachineLearning::getAccuracy()
		{
			int hits = 0;
			int total = 0;
			for (size_t i = 0; i < confusionMatrix.size(); i++)
			{
				for (size_t j = 0; j < confusionMatrix.at(i).size(); j++)
				{
					if (i == j) hits += confusionMatrix.at(i).at(j);
					total += confusionMatrix.at(i).at(j);
				}
			}
			return hits / (float)total;
		}

		/**
		* Save our obtained models (neural network, bag of words vocabulary
		* and class names) to use it later
		*/
		void MachineLearning::saveModels()
		{
			std::cout << "Saving model" << std::endl;
			mlp->save("mlp.yaml");
			cv::FileStorage fs("vocabulary.yaml", cv::FileStorage::WRITE);
			fs << "vocabulary" << vocabulary;
			fs.release();
			std::ofstream classesOutput("classes.txt");
			for (auto it = classes.begin(); it != classes.end(); ++it)
			{
				classesOutput << getClassId(classes, *it) << " " << *it << std::endl;
			}
			classesOutput.close();
		}

		void MachineLearning::loadModel()
		{
			std::cout << "Loading model" << std::endl;
			mlp = cv::ml::ANN_MLP::load("mlp.yaml");
			cv::FileStorage fsr("vocabulary.yaml", cv::FileStorage::READ);
			fsr["vocabulary"] >> vocabulary;
			fsr.release();
			std::ifstream infile("classes.txt");
			std::string temp;
			int temp2;
			std::map<int, std::string> classesMap;
			int countClasses = 0;
			while (infile >> temp2 >> temp)
			{
				countClasses++;
				classesMap[temp2] = temp;
			}
			while (countClasses)
			{
				classes.insert(classesMap[--countClasses]);
			}
			
			std::cout << "Training FLANN..." << std::endl;
			double start = cv::getTickCount();

			flann.add(vocabulary);
			flann.train();
			std::cout << "Time elapsed in minutes: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0 << std::endl;
		}




		void MachineLearning::train(std::string imgDir)
		{
			imagesDir = imgDir;
			train();
		}

		void MachineLearning::train()
		{
			double start = (double)cv::getTickCount();
			std::cout << "Reading training set..." << std::endl;
			files = getFilesInDirectory(imagesDir);
			std::random_shuffle(files.begin(), files.end());


			readImages(files.size()*trainSplitRatio, files.begin(), files.begin() + (size_t)(files.size() * trainSplitRatio),
				[&](const std::string& classname, const cv::Mat& descriptors) {
				// Append to the set of classes
				classes.insert(classname);
				// Append to the list of descriptors
				descriptorsSet.push_back(descriptors);
				// Append metadata to each extracted feature
				MachineLearning::MachineLearning::ImageData* data = new MachineLearning::MachineLearning::ImageData;
				data->classname = classname;
				data->bowFeatures = cv::Mat::zeros(cv::Size(networkInputSize, 1), CV_32F);
				for (int j = 0; j < descriptors.rows; j++)
				{
					descriptorsMetadata.push_back(data);
				}
			});
			std::cout << "Time elapsed in minutes: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0 << std::endl;

			std::cout << "Creating vocabulary..." << std::endl;
			start = (double)cv::getTickCount();

			// Use k-means to find k centroids (the words of our vocabulary)
			cv::kmeans(descriptorsSet, networkInputSize, labels, cv::TermCriteria(cv::TermCriteria::EPS +
				cv::TermCriteria::MAX_ITER, 10, 0.01), 1, cv::KMEANS_PP_CENTERS, vocabulary);
			// No need to keep it on memory anymore
			descriptorsSet.release();
			std::cout << "Time elapsed in minutes: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0 << std::endl;

			// using the bag of words technique
			std::cout << "Getting histograms of visual words..." << std::endl;
			int* ptrLabels = (int*)(labels.data);
			int size = labels.rows * labels.cols;
			for (int i = 0; i < size; i++)
			{
				int label = *ptrLabels++;
				MachineLearning::MachineLearning::ImageData* data = descriptorsMetadata[i];
				data->bowFeatures.at<float>(label)++;
			}

			// Filling matrixes to be used by the neural network
			std::cout << "Preparing neural network..." << std::endl;
			cv::Mat trainSamples;
			cv::Mat trainResponses;
			std::set<MachineLearning::MachineLearning::ImageData*> uniqueMetadata(descriptorsMetadata.begin(), descriptorsMetadata.end());
			for (auto it = uniqueMetadata.begin(); it != uniqueMetadata.end(); )
			{
				MachineLearning::MachineLearning::ImageData* data = *it;
				cv::Mat normalizedHist;
				cv::normalize(data->bowFeatures, normalizedHist, 0, data->bowFeatures.rows, cv::NORM_MINMAX, -1, cv::Mat());
				trainSamples.push_back(normalizedHist);
				trainResponses.push_back(getClassCode(classes, data->classname));
				delete *it; // clear memory
				it++;
			}
			descriptorsMetadata.clear();

			// Training neural network
			std::cout << "Training neural network..." << std::endl;
			start = cv::getTickCount();
			mlp = getTrainedNeuralNetwork(trainSamples, trainResponses);
			std::cout << "Time elapsed in minutes: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0 << std::endl;

			// We can clear memory now 
			trainSamples.release();
			trainResponses.release();
		
		
			// Train FLANN 
			std::cout << "Training FLANN..." << std::endl;
			start = cv::getTickCount();
			
			flann.add(vocabulary);
			flann.train();
			std::cout << "Time elapsed in minutes: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0 << std::endl;

		
		}

		void MachineLearning::testSet(std::string imgDir)
		{
			imagesDir = imgDir;
			testSet();
		}

		void MachineLearning::testSet()
		{
			// Reading test set 
			std::cout << "Reading test set..." << std::endl;
			double start = cv::getTickCount();
			if (files.size() == 0)
			{
				files = getFilesInDirectory(imagesDir);
				std::random_shuffle(files.begin(), files.end());
			}
			
			readImages(files.size()*(1.f - trainSplitRatio), files.begin() + (size_t)(files.size() * trainSplitRatio), files.end(),
				[&](const std::string& classname, const cv::Mat& descriptors) {
				// Get histogram of visual words using bag of words technique
				cv::Mat bowFeatures = getBOWFeatures(flann, descriptors, networkInputSize);
				cv::normalize(bowFeatures, bowFeatures, 0, bowFeatures.rows, cv::NORM_MINMAX, -1, cv::Mat());
				testSamples.push_back(bowFeatures);
				testOutputExpected.push_back(getClassId(classes, classname));
			});
			std::cout << "Time elapsed in minutes: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0 << std::endl;
		}








		int MachineLearning::testThis(int argc, char** argv)
		{
			/*if (argc != 4)
			{
				std::cerr << "Usage: <IMAGES_DIRECTORY>  <NETWORK_INPUT_LAYER_SIZE> <TRAIN_SPLIT_RATIO>" << std::endl;
				exit(-1);
			}
			std::string imagesDir = argv[1];
			int networkInputSize = atoi(argv[2]);
			float trainSplitRatio = atof(argv[3]);

			std::cout << "Reading training set..." << std::endl;
			double start = (double)cv::getTickCount();
			std::vector<std::string> files = getFilesInDirectory(imagesDir);
			std::random_shuffle(files.begin(), files.end());

			cv::Mat descriptorsSet;
			std::vector<ImageData*> descriptorsMetadata;
			std::set<std::string> classes;
			readImages(files.size(),files.begin(), files.begin() + (size_t)(files.size() * trainSplitRatio),
				[&](const std::string& classname, const cv::Mat& descriptors) {
				// Append to the set of classes
				classes.insert(classname);
				// Append to the list of descriptors
				descriptorsSet.push_back(descriptors);
				// Append metadata to each extracted feature
				ImageData* data = new ImageData;
				data->classname = classname;
				data->bowFeatures = cv::Mat::zeros(cv::Size(networkInputSize, 1), CV_32F);
				for (int j = 0; j < descriptors.rows; j++)
				{
					descriptorsMetadata.push_back(data);
				}
			});
			std::cout << "Time elapsed in minutes: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0 << std::endl;

			std::cout << "Creating vocabulary..." << std::endl;
			start = (double)cv::getTickCount();
			cv::Mat labels;
			cv::Mat vocabulary;
			// Use k-means to find k centroids (the words of our vocabulary)
			cv::kmeans(descriptorsSet, networkInputSize, labels, cv::TermCriteria(cv::TermCriteria::EPS +
				cv::TermCriteria::MAX_ITER, 10, 0.01), 1, cv::KMEANS_PP_CENTERS, vocabulary);
			// No need to keep it on memory anymore
			descriptorsSet.release();
			std::cout << "Time elapsed in minutes: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0 << std::endl;

			// Convert a set of local features for each image in a single descriptors
			// using the bag of words technique
			std::cout << "Getting histograms of visual words..." << std::endl;
			int* ptrLabels = (int*)(labels.data);
			int size = labels.rows * labels.cols;
			for (int i = 0; i < size; i++)
			{
				int label = *ptrLabels++;
				ImageData* data = descriptorsMetadata[i];
				data->bowFeatures.at<float>(label)++;
			}

			// Filling matrixes to be used by the neural network
			std::cout << "Preparing neural network..." << std::endl;
			cv::Mat trainSamples;
			cv::Mat trainResponses;
			std::set<ImageData*> uniqueMetadata(descriptorsMetadata.begin(), descriptorsMetadata.end());
			for (auto it = uniqueMetadata.begin(); it != uniqueMetadata.end(); )
			{
				ImageData* data = *it;
				cv::Mat normalizedHist;
				cv::normalize(data->bowFeatures, normalizedHist, 0, data->bowFeatures.rows, cv::NORM_MINMAX, -1, cv::Mat());
				trainSamples.push_back(normalizedHist);
				trainResponses.push_back(getClassCode(classes, data->classname));
				delete *it; // clear memory
				it++;
			}
			descriptorsMetadata.clear();

			// Training neural network
			std::cout << "Training neural network..." << std::endl;
			start = cv::getTickCount();
			cv::Ptr<cv::ml::ANN_MLP> mlp = getTrainedNeuralNetwork(trainSamples, trainResponses);
			std::cout << "Time elapsed in minutes: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0 << std::endl;

			// We can clear memory now 
			trainSamples.release();
			trainResponses.release();

			// Train FLANN 
			std::cout << "Training FLANN..." << std::endl;
			start = cv::getTickCount();
			cv::FlannBasedMatcher flann;
			flann.add(vocabulary);
			flann.train();
			std::cout << "Time elapsed in minutes: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0 << std::endl;

			// Reading test set 
			std::cout << "Reading test set..." << std::endl;
			start = cv::getTickCount();
			cv::Mat testSamples;
			std::vector<int> testOutputExpected;
			readImages(files.size(), files.begin() + (size_t)(files.size() * trainSplitRatio), files.end(),
				[&](const std::string& classname, const cv::Mat& descriptors) {
				// Get histogram of visual words using bag of words technique
				cv::Mat bowFeatures = getBOWFeatures(flann, descriptors, networkInputSize);
				cv::normalize(bowFeatures, bowFeatures, 0, bowFeatures.rows, cv::NORM_MINMAX, -1, cv::Mat());
				testSamples.push_back(bowFeatures);
				testOutputExpected.push_back(getClassId(classes, classname));
			});
			std::cout << "Time elapsed in minutes: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0 << std::endl;

			// Get confusion matrix of the test set
			std::vector<std::vector<int> > confusionMatrix = getConfusionMatrix();

			// Get accuracy of our model
			std::cout << "Confusion matrix: " << std::endl;
			printConfusionMatrix();
			std::cout << "Accuracy: " << getAccuracy() << std::endl;

			// Save models
			std::cout << "Saving models..." << std::endl;
			saveModels();
			*/
			return 0;
		}
	
}