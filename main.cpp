#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <functional>
#include <opencv2/opencv.hpp>


const int neighborhood_size = 3; //for masks
const double gauss_factor = 8;

const int hf_distance = 6;
const double alpha = 0.2;
const double v_border = 0.9;

const double a_max = 1.9; //limit contrast for small parts
const double a_new_max = 1.4; //a after limitation

const double flex_a = 1.;
const double flex_b = 1.;

//Dist
//OpenCV HSV: H [0-180], S [0-255], V [0-255]
const int gray_s = 19*255/100;
//const int gray_v = 51*255/100; //35
//const int gray_v = 100*255/100; 
const double inf_dist = 1000.;

int gray_v = 0;

struct BaseColor
{
	enum class Type {Colorful, Gray};

	int Hbegin;
	int Hend;
	int neighborhood_size;
	double gauss_factor;
	Type type;
};

//OpenCV HSV: H [0-180], S [0-255], V [0-255]
std::vector<BaseColor> baseColor{ //before transformation
	{30/2, 50/2, 2, 15, BaseColor::Type::Colorful}, //Brown
	{0, 0, 3, 8, BaseColor::Type::Gray},
	{50/2, 140/2, 3, 8, BaseColor::Type::Colorful}, //Green
	{185/2, 215/2, 1, 2, BaseColor::Type::Colorful}, //Shadows
};

std::vector<cv::Vec3b> color{ //we want to
	{17, 78, 62},
	{126, 112, 95},
	{5, 40, 40},
	{24, 23, 21},
};


std::vector<int> calcHist(const cv::Mat& img, int ch)
{
	std::vector<int> res(256, 0);
	for(int i = 0; i < img.rows; ++i) 
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
	for(int i = 0; i < img.rows; ++i) 
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

void apply(cv::Mat& img, cv::Mat& mask, std::function<cv::Vec3b(cv::Vec3b)> Fpixel)
{
	std::cout << "Applaying\n";
	for(int i = 0; i < img.rows; ++i) 
		for(int j = 0; j < img.cols; ++j)
		{
			cv::Vec3b p = img.at<cv::Vec3b>(i, j);
			double k = mask.at<cv::Vec3b>(i, j)[0] / 255.;
			cv::Vec3b newp = Fpixel(p);
			for(int ch = 0; ch < 3; ++ch)
			{
				img.at<cv::Vec3b>(i, j)[ch] = std::max(std::min(
							newp[ch] * k + p[ch] * (1 - k)
							, 255.), 0.); 
			}
		}
}

void contrast(cv::Mat& img, const cv::Mat& mask, cv::Vec3b mid, cv::Vec3b c, cv::Vec3b max, cv::Vec3b min, bool withMid = true)
{
	//Function constructoring
	std::vector<std::vector<double>> x {
		{min[0], max[0], mid[0]},	
		{min[1], max[1], mid[1]},	
		{min[2], max[2], mid[2]},	
	};
	std::vector<std::vector<double>> y {
		{     0,    255,   c[0]}, 	
		{     0,    255,   c[1]}, 	
		{     0,    255,   c[2]}, 	
	};
	
	cv::Mat A = cv::Mat_<double>(6, 6);
	for(int i = 0; i < 3; ++i)
	{
		A.at<double>(i, i) = 2*flex_a + std::accumulate(x[i].begin(), x[i].end(), 0, [](double sum, double xi){return sum + xi*xi;});
		A.at<double>(i, (i+1) % 3) = -flex_a;
		A.at<double>(i, (i+2) % 3) = -flex_a;
		A.at<double>(i, 3 + i) = std::accumulate(x[i].begin(), x[i].end(), 0);
		A.at<double>(i, 3 + (i+1) % 3) = 0;
		A.at<double>(i, 3 + (i+2) % 3) = 0;
	}

	for(int i = 0; i < 3; ++i)
	{
		A.at<double>(3 + i, i) = std::accumulate(x[i].begin(), x[i].end(), 0);
		A.at<double>(3 + i, (i+1) % 3) = 0;
		A.at<double>(3 + i, (i+2) % 3) = 0;
		A.at<double>(3 + i, 3 + i) = x[i].size() + 2*flex_b;
		A.at<double>(3 + i, 3 + (i+1) % 3) = -flex_b;
		A.at<double>(3 + i, 3 + (i+2) % 3) = -flex_b;
	}

	cv::Mat B = cv::Mat_<double>(6, 1);
   	for(int i = 0; i < 3; ++i)
	{
		double xy = 0;
		for(int j = 0; j < x[i].size(); ++j)
		{
			xy += x[i][j]*y[i][j];
		}
		B.at<double>(i, 0) = xy;
	}	
	for(int i = 0; i < 3; ++i)
	{
		B.at<double>(3 + i, 0) = std::accumulate(y[i].begin(), y[i].end(), 0); 
	}
	
	cv::Mat res;
	cv::solve(A, B, res, cv::DECOMP_LU); 
	for(int i = 0; i < res.rows; ++i)
	{
		std::cout << res.at<double>(i, 0) << " ";
	}
	std::cout << "\n";

//	auto Fpixel = [=](int x) { return a*x + b; };
	//Applying
//	apply(img, Fpixel);
}
	

double dist(cv::Vec3b a, BaseColor b)
{
	//Converting a to HSV
	cv::Mat ma(1, 1, CV_8UC3);
	ma.at<cv::Vec3b>(0, 0) = a;
	cv::cvtColor(ma, ma, CV_BGR2HSV);
	auto ha = ma.at<cv::Vec3b>(0, 0);
	
	BaseColor::Type type;
	if(ha[1] < gray_s && ha[2] > gray_v)
		type = BaseColor::Type::Gray;
	else
		type = BaseColor::Type::Colorful;
	if(type == b.type)
	{
		if(type == BaseColor::Type::Colorful)
		{
			if( ha[0] > b.Hbegin && ha[0] < b.Hend)
				return 0.0;
			else
				return inf_dist;
		}
		else
			return 0.0;
	}
	else
		return inf_dist;
}

void autoContrast(cv::Mat& image)
{
	//Gray_v calculation
	std::cout << "Gray_v calculation\n"; 
	{
		int v_max = 0;
		cv::Mat image_hsv;
		cv::cvtColor(image, image_hsv, CV_BGR2HSV);
		std::vector<int> h_freq(256, 0);
		for(int i = 0; i < image_hsv.rows; ++i)
			for(int j = 0; j < image_hsv.cols; ++j)
			{
				int v = image_hsv.at<cv::Vec3b>(i, j)[2];
				h_freq[v]++;
			}
		
		//for(int i = 0; i < h_freq.size(); ++i)
		//{
		//	std::cout << i << ": " << h_freq[i] << "\n";
		//}
		
		//Last peak detection
		for(int i = h_freq.size() - 1; i >= hf_distance; --i)
		{
			if(h_freq[i] - h_freq[i - hf_distance] > alpha*image.cols*image.rows/h_freq.size())
			{
				gray_v = i;
				break;
			}
		}
		gray_v *= v_border;
		std::cout << "gray_v: " << gray_v << "\n";
	}
	//Making masks
	std::cout << "Making masks\n";
	std::vector<cv::Mat> masks(baseColor.size());
	for(auto& mask: masks)
		mask = cv::Mat(image.rows, image.cols, CV_8UC3, cv::Scalar(0,0,0));
	cv::Mat outerMask(image.rows, image.cols, CV_8UC3, cv::Scalar(0,0,0));
	{
		for(int i = 0; i < image.rows; ++i) 
			for(int j = 0; j < image.cols; ++j)
			{	
				auto p = image.at<cv::Vec3b>(i, j);
				int closer = -1;
				for(int i = 0; i < baseColor.size(); ++i)
				{
					if(dist(p, baseColor[i]) < inf_dist)
						closer = i;
				}
				if(closer != -1)
					masks[closer].at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);
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

			for(int k = 0; k < image.rows; ++k) 
				for(int l = 0; l < image.cols; ++l)
					if(masks[i].at<cv::Vec3b>(k, l)[0] != 0)
						paint(k, l);
			//Blur
			cv::GaussianBlur(mask, masks[i], cv::Size(baseColor[i].neighborhood_size*baseColor[i].gauss_factor*2+1, baseColor[i].neighborhood_size*baseColor[i].gauss_factor*2+1), 0);
		}

		for(auto& mask: masks)
		{	
			cv::namedWindow( "Display window", CV_WINDOW_AUTOSIZE );// Create a window for display.
			cv::imshow( "Display window", mask );                   // Show our image inside it.
			cv::waitKey(0); 
		}

		//OuterMask generation
		for(int i = 0; i < outerMask.rows; ++i)
			for(int j = 0; j < outerMask.cols; ++j)
			{
				int sum = 0;
				for(auto& mask: masks)
					sum += mask.at<cv::Vec3b>(i, j)[0];
				outerMask.at<cv::Vec3b>(i, j) = cv::Vec3b(std::max(0, 255 - sum), std::max(0, 255 - sum), std::max(0, 255 - sum));
			}

		cv::namedWindow( "Display window", CV_WINDOW_AUTOSIZE );// Create a window for display.
		cv::imshow( "Display window", outerMask );                   // Show our image inside it.
		cv::waitKey(0); 


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
	auto t = getForMask(image, outerMask);
	auto mid = std::get<0>(t);
	auto min = std::get<1>(t);
	auto max = std::get<2>(t);
	contrast(image, outerMask, mid, mid, max, min, false);		
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
