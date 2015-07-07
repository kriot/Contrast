#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <opencv2/opencv.hpp>


const int distance = 6;
const double diff_factor = 0.01;
const int neighborhood_size = 6; //for masks
const double gauss_factor = 4;
const int hist_scale = 2;

const double alpha = 0.001; //in place of diff factor

std::vector<cv::Vec3b> baseColor{ //before transformation
	{39, 44, 40}, //green
	{76, 83, 90}, //gray
	{53, 53, 44}, //brown
};

std::vector<cv::Vec3b> color{ //we want to
	{12, 169, 12}, 
	{126, 112, 95},
	{17, 78, 62},
};


std::vector<int> calcHist(const cv::Mat& img, int ch)
{
	std::vector<int> res(256, 0);
	for(int i = 0; i < img.rows - 1; ++i) //Awful thing. Because the last row doesnt work correctly (seg.fault)
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
	auto f = [=](int x) { return a*x + b; };
	//Applying
	for(int i = 0; i < img.rows - 1; ++i) //Awful thing
		for(int j = 0; j < img.cols; ++j)
			for(int ch = 0; ch < 3; ++ch)
			{
				int v = img.at<cv::Vec3b>(i, j)[ch];
				double k = mask.at<cv::Vec3b>(i, j)[0] / 255.;
				img.at<cv::Vec3b>(i, j)[ch] = std::max(std::min(
							f(v) * k + v * (1 - k)
							, 255.), 0.); 
			}
	
}

void autoContrast(cv::Mat& image)
{
	auto comp = [](cv::Vec3b a, cv::Vec3b b){
		if(a[0] != b[0])
			return a[0] < b[0];
		if(a[1] != b[1])
			return a[1] < b[1];
		if(a[2] != b[2])
			return a[2] < b[2];
		return false;	
	};
	std::vector<std::set<cv::Vec3b, decltype(comp)>> freqColor(baseColor.size(), std::set<cv::Vec3b, decltype(comp)>(comp));
	//Base colors
	{
		//Calculatioing histogram (3D)
		std::cout << "Calculating histogram\n";
		std::vector<std::vector<std::vector<int>>> hist(256/hist_scale, 
				std::vector<std::vector<int>>(256/hist_scale, 
					std::vector<int>(256/hist_scale, 
						0)));
		for(int i = 0; i < image.rows; ++i)
			for(int j = 0; j < image.cols; ++j)
			{
				auto c = image.at<cv::Vec3b>(i, j);
				hist[c[0]/hist_scale][c[1]/hist_scale][c[2]/hist_scale]++;
			}

		//Choosing the peaks
		std::cout << "Peaks\n";
		long long s = image.rows * image.cols/pow(hist.size(), 3); //image area
		std::vector<cv::Vec3b> peaks;
		for(int i = 0; i < hist.size(); ++i)
			for(int j = 0; j < hist[i].size(); ++j)
				for(int k = 0; k < hist[i][j].size(); ++k)
				{
					auto check_point = [&](int i1, int j1, int k1) 
					{
						return 
							0 < i1 && i1 < 256 &&
							0 < j1 && j1 < 256 &&
							0 < k1 && k1 < 256 &&
							hist[i1][j1][k1] < hist[i][j][k] - s*diff_factor;
					};
/*					if(check_point(i - distance, j - distance, k - distance) &&
					   check_point(i - distance, j - distance, k + distance) &&
					   check_point(i - distance, j + distance, k - distance) &&
					   check_point(i - distance, j + distance, k + distance) &&
					   check_point(i + distance, j - distance, k - distance) &&
					   check_point(i + distance, j - distance, k + distance) &&
					   check_point(i + distance, j + distance, k - distance) &&
					   check_point(i + distance, j + distance, k + distance))*/
					if(hist[i][j][k] > s*alpha)
					{
						peaks.push_back(cv::Vec3b(i, j, k));
					}
				}
		//Classification
		for(cv::Vec3b p: peaks)
		{
			auto dist = [](cv::Vec3b a, cv::Vec3b b)
			{
				cv::Mat ma(1, 1, CV_8UC3);
				ma.at<cv::Vec3b>(0, 0) = a;
				cv::cvtColor(ma, ma, CV_RGB2HSV);
				cv::Mat mb(1, 1, CV_8UC3);
				mb.at<cv::Vec3b>(0, 0) = b;
				cv::cvtColor(mb, mb, CV_RGB2HSV);
			//	return pow(ma.at<cv::Vec3b>(0, 0)[0] - mb.at<cv::Vec3b>(0, 0)[0], 2);
				return pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2) + pow(a[2] - b[2], 2);
			};
			int closer = 0;
			for(int i = 0; i < baseColor.size(); ++i)
			{
				if(dist(p, baseColor[i]) < dist(p, baseColor[closer]))
					closer = i;
			}
			freqColor[closer].insert(p);
		}
		for(int i = 0; i < freqColor.size(); ++i)
		{
			std::cout << "Color " << i << ":\n";
			for(auto p: freqColor[i])
				std::cout << "Peak: " << (int)p[0] << ", " << (int)p[1] << ", " << (int)p[2] <<"\n";
		}
	}
	//Making masks
	std::cout << "Making masks\n";
	std::vector<cv::Mat> masks(freqColor.size());
	for(auto& mask: masks)
		mask = cv::Mat(image.rows, image.cols, CV_8UC3, cv::Scalar(0,0,0));
	{
		for(int i = 0; i < image.rows - 1; ++i) //Awful thing
			for(int j = 0; j < image.cols; ++j)
			{	
				auto p = image.at<cv::Vec3b>(i, j);
				int color = -1;
				for(int i = 0; i < freqColor.size() && color == -1; ++i)
					if(freqColor[i].find(p) != freqColor[i].end())
						color = i;
				if(color != -1)
					masks[color].at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);
			}

		for(auto& mask: masks)
		{	
			cv::namedWindow( "Display window", CV_WINDOW_AUTOSIZE );// Create a window for display.
			cv::imshow( "Display window", mask );                   // Show our image inside it.
			cv::waitKey(0); 
		}


		//Inflating masks and blur
		std::cout << "Improving masks\n";

		for(int i = 0; i < masks.size(); ++i)
		{
			auto mask = cv::Mat(image.rows, image.cols, CV_8UC3, cv::Scalar(0,0,0));
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
			cv::GaussianBlur(mask, masks[i], cv::Size(neighborhood_size*gauss_factor*2+1, neighborhood_size*gauss_factor*2+1), 0);
		}

	}
	//Contrast
	std::cout << "Contrast processing\n";
	for(int i = 0; i < masks.size(); ++i)
	{
		auto t = getForMask(image, masks[i]);
		auto mid = std::get<0>(t);
		auto min = std::get<1>(t);
		auto max = std::get<2>(t);
		contrast(image, masks[i], mid, color[i], max, min);		
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
	
	//Processing
	autoContrast(image);

	//Out
	cv::namedWindow( "Display window", CV_WINDOW_AUTOSIZE );// Create a window for display.
	cv::imshow( "Display window", image );                   // Show our image inside it.
	cv::waitKey(0); 
	
	cv::imwrite( "output.tif", image);
}
