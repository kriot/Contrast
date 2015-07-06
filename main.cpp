#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>


std::vector<long long> calcHist(const cv::Mat& img, int ch)
{
	std::vector<long long> res(256, 0);
	for(int i = 0; i < img.rows - 1; ++i) //Awful thing
		for(int j = 0; j < img.cols; ++j)
		{
			auto t = img.at<cv::Vec4b>(i, j); 
			res[img.at<cv::Vec4b>(i, j)[ch]]++;
		}
	return res;
}

int main(int argc, char** argv)
{
	if(argc == 1)
	{
		std::cerr << "Image name is needed\n";
		return -1;
	}
	cv::Mat image = cv::imread(argv[1]);
	if(!image.data)
	{
		std::cerr << "Cant read image\n";
		return -1;
	}
	std::cout << "Image is read\n";
	std::vector<std::vector<long long>> hist = {calcHist(image, 1), calcHist(image, 2), calcHist(image, 3)};
	for(auto h: hist[0])
		std::cout << h << " ";
}
