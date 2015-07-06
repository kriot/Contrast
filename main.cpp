#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <opencv2/opencv.hpp>


const int distance = 6;
const double diff_factor = 0.01;

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

	// Printing hist
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
			if((i-distance > 0 && h[i-distance] < h[i] - s*diff_factor) && 
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
	//Making masks
	double b = 0;
	for(auto _b: peaks[0])
	{
		b += _b;
	}
	b /= peaks[0].size();
	double g = 0;
	for(auto _g: peaks[1])
	{
		g += _g;
	}
	g /= peaks[1].size();
	double r = 0;
	for(auto _r: peaks[2])
	{
		r += _r;
	}
	r /= peaks[2].size();

	std::vector<cv::Mat> masks(9);
	for(auto& mask: masks)
		mask = cv::Mat(image.rows, image.cols, CV_8UC3, cv::Scalar(70,70,70));
	for(int i = 0; i < image.rows - 1; ++i) //Awful thing
		for(int j = 0; j < image.cols; ++j)
		{	
			auto p = image.at<cv::Vec3b>(i, j);
			/*
			auto isThere = [&](int v, int c) { return peaks[c].find(v) != peaks[c].end(); };
			int color = isThere(p[0], 0) << 0 | isThere(p[1], 1) << 1 | isThere(p[2], 2) << 2;
			masks[color].at<cv::Vec3b>(i, j) = cv::Vec3b(isThere(p[0],0)*255, isThere(p[1],1)*255, isThere(p[2],2)*255);
			*/
			//another way
			std::vector<std::tuple<int, int, int>> defColors({/*std::make_tuple(0,0,0),*/ std::make_tuple(0, 0, r), std::make_tuple(0, g, 0), /*std::make_tuple(0, g, r),*/ std::make_tuple(b, 0, 0), /*std::make_tuple(b, 0, r),*/ /*std::make_tuple(b, g, 0), std::make_tuple(b, g, r)*/});
			std::vector<int> dist;
			for(auto c: defColors)
				dist.push_back((std::get<0>(c)-p[0])*(std::get<0>(c)-p[0])+(std::get<1>(c)-p[1])*(std::get<1>(c)-p[1])+(std::get<2>(c)-p[2])*(std::get<2>(c)-p[2]));
			int color = 0;
//			std::cout << "Dists:\n";
			for(int i = 0; i < defColors.size(); ++i)
			{
//				std::cout << dist[i] << " ";
				if(dist[color] > dist[i])
					color = i;
			}
//			std::cout << "\n";
			masks[color].at<cv::Vec3b>(i, j) = cv::Vec3b(std::get<0>(defColors[color]), std::get<1>(defColors[color]), std::get<2>(defColors[color]));
		}

	for(int i = 0; i < 9; ++i)
	{	
		cv::namedWindow( "Display window", CV_WINDOW_AUTOSIZE );// Create a window for display.
		cv::imshow( "Display window", masks[i] );                   // Show our image inside it.
		cv::waitKey(0); 
	}
}
