#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <opencv2/opencv.hpp>

const double alpha = 0.1/255;

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

std::tuple<cv::Vec3b, cv::Vec3b, cv::Vec3b> getForMask(cv::Mat& img, const cv::Mat& mask) //mid, min, max
{
	cv::Vec3b max(0,0,0);
	cv::Vec3b min(255,255,255);
	long long count = 0;
	std::vector<long long> mid{0, 0, 0};
	for(int i = 0; i < img.rows - 1; ++i) //Awful thing
		for(int j = 0; j < img.cols; ++j)
			if(mask.at<cv::Vec3b>(i, j)[0] > 0)
			{
				auto p = img.at<cv::Vec3b>(i, j);
				
				mid[0] += p[0];
				mid[1] += p[1];
				mid[2] += p[2];
				count++;

				for(int i = 0; i < 3; ++i)
				{
					if(p[i] > max[i]) 
						max[i] = p[i];
					if(p[i] < min[i])
						min[i] = p[i];
				}
			}

	return std::make_tuple(cv::Vec3b(mid[0]/(count+1), mid[1]/(count+1), mid[2]/(count+1)), min, max);
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

	int begin = 0, end = 255;

	auto y = [=](int begin){ return 0.11*hist[0][begin] + 0.59*hist[1][begin] + 0.3*hist[2][begin]; };
	while(y(begin) < alpha*image.cols*image.rows && begin < end - 1)
		begin++;
	while(y(end) < alpha*image.cols*image.rows && begin < end - 1)
		end--;

	//Contrast
	auto f = [=](int x) { return std::min(std::max((x - begin) * 255 / (end - begin), 0), 255); };
	//Applying
	for(int i = 0; i < image.rows - 1; ++i) //Awful thing
		for(int j = 0; j < image.cols; ++j)
			for(int ch = 0; ch < 3; ++ch)
			{
				int v = image.at<cv::Vec3b>(i, j)[ch];
				double k = 1;
				image.at<cv::Vec3b>(i, j)[ch] = std::max(std::min(
							f(v) * k + v * (1 - k)
							, 255.), 0.); 
			}

	
	cv::namedWindow( "Display window", CV_WINDOW_AUTOSIZE );// Create a window for display.
	cv::imshow( "Display window", image );                   // Show our image inside it.
	cv::waitKey(0); 
	
	cv::imwrite( "output.tif", image);
}
