#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <opencv2/opencv.hpp>


const int distance = 3;
const double diff_factor = 0.001;

std::vector<long long> calcHist(const cv::Mat& img, int ch)
{
	std::vector<long long> res(256, 0);
	for(int i = 0; i < img.rows - 1; ++i) //Awful thing
		for(int j = 0; j < img.cols; ++j)
		{
			res[img.at<cv::Vec3b>(i, j)[ch]]++;
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
	cv::Mat image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
	if(!image.data)
	{
		std::cerr << "Cant read image\n";
		return -1;
	}
	std::cout << "Image is read\n";
	//Calculatioing histograms
	std::vector<std::vector<long long>> hist = {calcHist(image, 0), calcHist(image, 1), calcHist(image, 2)};
	std::cout << "Hists are done\n";
	for(int ch = 0; ch < 3; ++ch)
	{
		for(auto v: hist[ch])
			std::cout << v << " ";
		std::cout << "\n";
	}
	//Choosing the peaks
	long long s = image.rows * image.cols;
	std::vector<std::set<int>> peaks;
	for(auto h: hist)
	{
		peaks.push_back(std::set<int>());
		for(int i = 0; i < 256; ++i)
		{
			if((i-distance > 0 && h[i-distance] < h[i] - s*diff_factor) || 
			   (i+distance < 256 && h[i+distance] < h[i] - s*diff_factor))
			{
				peaks.back().insert(i);
			}
		}
	}
	for(auto b: peaks)
	{
		for(auto color_ch: b)
			std::cout << color_ch << " ";
		std::cout << "\n";
	}
}
