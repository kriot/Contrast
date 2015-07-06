#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <opencv2/opencv.hpp>

const double alpha = 0.01/255;
const int kernel_size = 20;

std::vector<long long> calcHist(const cv::Mat& img, int ch, int i0, int j0, int i1, int j1)
{
	std::vector<long long> res(256, 0);
	for(int i = i0; i < i1; ++i) 
		for(int j = j0; j < j1; ++j)
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


	//Contrast
	for(int i = 0; i < image.rows - 1; ++i) //Awful thing
		for(int j = 0; j < image.cols; ++j)
			for(int ch = 0; ch < 3; ++ch)
			{
				int begin = 0, end = 255;
				std::vector<std::vector<long long>> hist = {
					calcHist(image, 0, std::max(i - kernel_size, 0), std::max(j - kernel_size, 0), std::min(i + kernel_size, image.rows), std::min(j + kernel_size, image.cols)), 
					calcHist(image, 1, std::max(i - kernel_size, 0), std::max(j - kernel_size, 0), std::min(i + kernel_size, image.rows), std::min(j + kernel_size, image.cols)), 
					calcHist(image, 2, std::max(i - kernel_size, 0), std::max(j - kernel_size, 0), std::min(i + kernel_size, image.rows), std::min(j + kernel_size, image.cols))};

				auto y = [=](int begin){ return 0.11*hist[0][begin] + 0.59*hist[1][begin] + 0.3*hist[2][begin]; };
				while(y(begin) < alpha*kernel_size*kernel_size && begin < end - 1)
					begin++;
				while(y(end) < alpha*kernel_size*kernel_size && begin < end - 1)
					end--;

				auto f = [&](int x) { return std::min(std::max((x - begin) * 255 / (end - begin), 0), 255); };

				int v = image.at<cv::Vec3b>(i, j)[ch];
				image.at<cv::Vec3b>(i, j)[ch] = f(v);
			}

	
	cv::namedWindow( "Display window", CV_WINDOW_AUTOSIZE );// Create a window for display.
	cv::imshow( "Display window", image );                   // Show our image inside it.
	cv::waitKey(0); 
	
	cv::imwrite( "output.tif", image);
}
