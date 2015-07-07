#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <opencv2/opencv.hpp>


const int distance = 6;
const double diff_factor = 0.01;
const int neighborhood_size = 10; //for masks
const double gauss_factor = 2;
std::vector<cv::Vec3b> color{
	{0, 0, 0}, 
	{0, 0, 150},
	{0, 250, 0},
	{34, 250, 250},
	{203, 104, 0},
	{200, 0, 200},
	{0, 200, 0},
	{200, 200, 200},
	
};

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

cv::Mat processed;

void contrast(cv::Mat& img, const cv::Mat& mask, cv::Vec3b mid, cv::Vec3b c, cv::Vec3b max, cv::Vec3b min)
{
	//Function constructoring
	int max_cord = std::max({max[0], max[1], max[2]});
	int min_cord = std::min({min[0], min[1], min[2]});
	std::vector<double> x{min_cord, max_cord, mid[0], mid[1], mid[2]};
	std::cout << "Min: " << min_cord << ", max: " << max_cord << "\n";
	std::vector<double> y{       0,      255,   c[0],   c[1],   c[2]};
	double sumx = 0,
		   sumy = 0,
		   sumxp = 0,
		   sumxy = 0;
	for(int i = 0; i < x.size(); ++i)
	{
		sumx  += x[i];
		sumy  += y[i];
		sumxp += x[i]*x[i];
		sumxy += x[i]*y[i];
	}
	double a = (sumx*sumy - x.size()*sumxy) / (sumx*sumx - x.size()*sumxp);
	double b = (sumx*sumxy - sumxp*sumy)    / (sumx*sumx - x.size()*sumxp);
	std::cout << "A: " << a << ", b: " << b <<"\n";
	if (a < 1)
		return;
	b -= std::abs(b)/2;
	auto f = [=](int x) { return a*x + b; };
	//Applying
	for(int i = 0; i < img.rows - 1; ++i) //Awful thing
		for(int j = 0; j < img.cols; ++j)
			if(true||processed.at<cv::Vec3b>(i, j)[0] == 0)
				for(int ch = 0; ch < 3; ++ch)
				{
					int v = img.at<cv::Vec3b>(i, j)[ch];
					double k = mask.at<cv::Vec3b>(i, j)[0] / 255.;
					img.at<cv::Vec3b>(i, j)[ch] = std::max(std::min(
								f(v) * k + v * (1 - k)
								, 255.), 0.); 
					processed.at<cv::Vec3b>(i, j)[0] = 100;
				}
	
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
	processed = cv::Mat(image.rows, image.cols, CV_8UC3, cv::Scalar(0,0,0));
	//Calculatioing histograms
	std::vector<std::vector<int>> hist = {calcHist(image, 0), calcHist(image, 1), calcHist(image, 2)};
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
	std::cout << "Making masks\n";
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
		mask = cv::Mat(image.rows, image.cols, CV_8UC3, cv::Scalar(0,0,0));
	for(int i = 0; i < image.rows - 1; ++i) //Awful thing
		for(int j = 0; j < image.cols; ++j)
		{	
			auto p = image.at<cv::Vec3b>(i, j);
//			auto isThere = [&](int v, int c) { return peaks[c].find(v) != peaks[c].end(); };
//			int color = isThere(p[0], 0) << 0 | isThere(p[1], 1) << 1 | isThere(p[2], 2) << 2;
//			masks[color].at<cv::Vec3b>(i, j) = cv::Vec3b(isThere(p[0],0)*255, isThere(p[1],1)*255, isThere(p[2],2)*255);
			//another way
			
			std::vector<std::tuple<int, int, int>> defColors({std::make_tuple(-10,-10,-10), std::make_tuple(-10, -10, r), std::make_tuple(-10, g, -10), std::make_tuple(0, 255, 255), std::make_tuple(b, -10, -10), std::make_tuple(255, 0, 255), std::make_tuple(255, 255, 0), std::make_tuple(255, 255, 255)});
			std::vector<int> dist;
			for(auto c: defColors)
				dist.push_back((std::get<0>(c)-p[0])*(std::get<0>(c)-p[0])+(std::get<1>(c)-p[1])*(std::get<1>(c)-p[1])+(std::get<2>(c)-p[2])*(std::get<2>(c)-p[2]));
			int color = 0;
			for(int i = 0; i < defColors.size(); ++i)
				if(dist[color] > dist[i])
					color = i;

			//colorful
			//masks[color].at<cv::Vec3b>(i, j) = cv::Vec3b(std::get<0>(defColors[color]), std::get<1>(defColors[color]), std::get<2>(defColors[color]));
			//for work
			masks[color].at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);
		}

	//Inflating masks and blur
	std::cout << "Improving masks\n";
	std::vector<cv::Mat> nmasks(masks.size());
	
	for(int i = 0; i < masks.size(); ++i)
	{
		auto mask = cv::Mat(image.rows, image.cols, CV_8UC3, cv::Scalar(0,0,0));
		nmasks[i] = cv::Mat(image.rows, image.cols, CV_8UC3, cv::Scalar(0,0,0));
		//Inflating
		auto paint = [&] (int i0, int j0) {
			for(int k = -neighborhood_size; k <= neighborhood_size; ++k)
				for(int l = -neighborhood_size; l <= neighborhood_size; ++l)
					if(i0+k >= 0 && i0+k < image.rows && j0+l >=0 && j0+l < image.cols)
						mask.at<cv::Vec3b>(i0+k, j0+l) = cv::Vec3b(255,255,255);
		};

		for(int k = 0; k < image.rows - 1; ++k) //Awful thing
			for(int l = 0; l < image.cols; ++l)
				if(masks[i].at<cv::Vec3b>(k, l)[0] != 0)
					paint(k, l);
		//Blur
//		mask.copyTo(nmasks[i]);
		cv::GaussianBlur(mask, nmasks[i], cv::Size(neighborhood_size*gauss_factor*2+1, neighborhood_size*gauss_factor*2+1), 0);
	}

	for(int i = 0; i < 9; ++i)
	{	
		cv::namedWindow( "Display window", CV_WINDOW_AUTOSIZE );// Create a window for display.
		cv::imshow( "Display window", nmasks[i] );                   // Show our image inside it.
		cv::waitKey(0); 
	}

	//Contrast
	std::cout << "Contrast processing\n";
	for(int i = 0; i < nmasks.size(); ++i)
	{
		auto t = getForMask(image, nmasks[i]);
		auto mid = std::get<0>(t);
		auto min = std::get<1>(t);
		auto max = std::get<2>(t);
		auto middle = [&] (cv::Vec3b m) {return cv::Vec3b((m[0] + b)/2, (m[1] + g)/2, (m[2] + r)/2);};
		contrast(image, nmasks[i], mid, color[i], max, min);		
	}
	cv::namedWindow( "Display window", CV_WINDOW_AUTOSIZE );// Create a window for display.
	cv::imshow( "Display window", image );                   // Show our image inside it.
	cv::waitKey(0); 
	
	cv::imwrite( "output.tif", image);
}
