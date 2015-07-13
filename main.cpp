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

const int minmax_distance = 6;
const double minmax_alpha = 0.2;
const int minmax_border = 1.2;

const int mask_balance = 100;

const double flex_a = 100.;
const double flex_b = 100.;

//Dist
//OpenCV HSV: H [0-180], S [0-255], V [0-255]
const int gray_s = 19*255/100;
//const int gray_v = 51*255/100; //35
//const int gray_v = 100*255/100; 
const double inf_dist = 1000.;

const std::vector<double> default_prior = {1., 1., 1.};

int gray_v = 0;

struct BaseColor
{
	enum class Type {Colorful, Gray};

	int Hbegin;
	int Hend;
	int neighborhood_size;
	double gauss_factor;
	Type type;
	std::vector<double> prior; //min, max, color
};

//OpenCV HSV: H [0-180], S [0-255], V [0-255]
std::vector<BaseColor> baseColor{ //before transformation
	{30/2, 50/2, 2, 15, BaseColor::Type::Colorful, {1., 1., 10.}}, //Brown
	{0, 0, 3, 8, BaseColor::Type::Gray, {1., 1., 1.}},
	{50/2, 140/2, 3, 8, BaseColor::Type::Colorful, {1., 1., 500.}}, //Green
//	{185/2, 215/2, 1, 2, BaseColor::Type::Colorful}, //Shadows
};

std::vector<cv::Vec3b> color{ //we want to
	{17, 78, 62},
	{126, 112, 95},
	{5, 40, 40},
//	{24, 23, 21},
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
	std::vector<std::vector<int>> hist(3, std::vector<int>(255, 0));

	for(int i = 0; i < img.rows; ++i) 
		for(int j = 0; j < img.cols; ++j)
			if(mask.at<cv::Vec3b>(i, j)[0] > mask_balance)
			{
				auto p = img.at<cv::Vec3b>(i, j);
				
				mid[0] += p[0];
				mid[1] += p[1];
				mid[2] += p[2];
				count++;
				
				for(int ch = 0; ch < 3; ++ch)
					hist[ch][p[ch]]++;
			}

	//Peaks
	for(int ch = 0; ch < 3; ++ch)
	{
		//Min
		for(int i = 0; i < hist[ch].size() - minmax_distance; ++i)
		{
			if(hist[ch][i] - hist[ch][i + minmax_distance] > minmax_alpha*img.cols*img.rows/hist[ch].size())
			{
				min[ch] = i;
				min[ch] *= minmax_border;
				break;
			}
		}
		
		//Max
		for(int i = hist[ch].size() - 1; i >= minmax_distance; --i)
		{
			if(hist[ch][i] - hist[ch][i - minmax_distance] > minmax_alpha*img.cols*img.rows/hist[ch].size())
			{
				max[ch] = i;
				max[ch] *= minmax_border;
				break;
			}
		}
	}

	return std::make_tuple(cv::Vec3b(mid[0]/(count+1), mid[1]/(count+1), mid[2]/(count+1)), min, max);
}

void apply(cv::Mat& img, const cv::Mat& mask, std::function<cv::Vec3b(cv::Vec3b)> Fpixel)
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
				img.at<cv::Vec3b>(i, j)[ch] = newp[ch] * k + p[ch] * (1 - k);
			}
		}
}

void contrast(cv::Mat& img, const cv::Mat& mask, cv::Vec3b mid, cv::Vec3b c, cv::Vec3b max, cv::Vec3b min, std::vector<double> prior, bool withMid = true)
{
	//Function constructoring
	std::vector<std::vector<double>> x;
	std::vector<std::vector<double>> y;
	if(withMid)
	{
		x = {
			{min[0], max[0], mid[0]},	
			{min[1], max[1], mid[1]},	
			{min[2], max[2], mid[2]},
		};
		y = {
			{     0,    255,   c[0]}, 	
			{     0,    255,   c[1]}, 	
			{     0,    255,   c[2]}, 	
		};
	}
	else
	{
		x = {
			{min[0], max[0]},	
			{min[1], max[1]},	
			{min[2], max[2]},
		};
		y = {
			{     0,    255}, 	
			{     0,    255}, 	
			{     0,    255}, 	
		};
	}

	std::cout << "Min: " << (int)min[0] << " " << (int)min[1] << " " << (int)min[2] << "\n";
	std::cout << "Max: " << (int)max[0] << " " << (int)max[1] << " " << (int)max[2] << "\n";
	std::cout << "Mid: " << (int)mid[0] << " " << (int)mid[1] << " " << (int)mid[2] << "\n";
	
	//Cols: a1, a2, a3, b1, b2, b3
	//Rows for: da1, da2, da3, db1, db2, db3
	cv::Mat A = cv::Mat_<double>(6, 6);
	for(int i = 0; i < 3; ++i)
	{
		{
			double k = 0;
			for(int j = 0; j < x[i].size(); ++j)
				k += prior[j]*x[i][j]*x[i][j];
			A.at<double>(i, i) = 2*flex_a + k;
		}
		A.at<double>(i, (i+1) % 3) = -flex_a;
		A.at<double>(i, (i+2) % 3) = -flex_a;
		{
			double k = 0;
			for(int j = 0; j < x[i].size(); ++j)
				k += prior[j]*x[i][j]; 
			A.at<double>(i, 3 + i) = k;
		}
		A.at<double>(i, 3 + (i+1) % 3) = 0;
		A.at<double>(i, 3 + (i+2) % 3) = 0;
	}

	for(int i = 0; i < 3; ++i)
	{
		{
			double k = 0;
			for(int j = 0; j < x[i].size(); ++j)
				k += prior[j]*x[i][j];
			A.at<double>(3 + i, i) = k;
		}
		A.at<double>(3 + i, (i+1) % 3) = 0;
		A.at<double>(3 + i, (i+2) % 3) = 0;
		{
			double k = 0;
			for(int j = 0; j < x[i].size(); ++j)
				k += prior[j];
			A.at<double>(3 + i, 3 + i) = k + 2*flex_b;
		}
		A.at<double>(3 + i, 3 + (i+1) % 3) = -flex_b;
		A.at<double>(3 + i, 3 + (i+2) % 3) = -flex_b;
	}

	cv::Mat B = cv::Mat_<double>(6, 1);
   	for(int i = 0; i < 3; ++i)
	{
		double xy = 0;
		for(int j = 0; j < x[i].size(); ++j)
		{
			xy += x[i][j]*y[i][j]*prior[j];
		}
		B.at<double>(i, 0) = xy;
	}	
	for(int i = 0; i < 3; ++i)
	{
		double k = 0;
		for(int j = 0; j < x[i].size(); ++j)
			k += prior[j]*y[i][j];
		B.at<double>(3 + i, 0) = k; 
	}
	
	cv::Mat res;
	cv::solve(A, B, res, cv::DECOMP_LU); 

	std::cout << "Params: ";
	for(int i = 0; i < res.rows; ++i)
	{
		std::cout << res.at<double>(i, 0) << " ";
	}
	std::cout << "\n";

	std::function<cv::Vec3b(cv::Vec3b)> Fpixel = [=](cv::Vec3b x) mutable -> cv::Vec3b { 
		cv::Vec3b r(0, 0, 0);
		r[0] = std::max(std::min( res.at<double>(0, 0)*x[0] + res.at<double>(3 + 0, 0), 255.), 0.); 		
		r[1] = std::max(std::min( res.at<double>(1, 0)*x[1] + res.at<double>(3 + 1, 0), 255.), 0.); 		
		r[2] = std::max(std::min( res.at<double>(2, 0)*x[2] + res.at<double>(3 + 2, 0), 255.), 0.);
		return r;
	};
	//Applying
	apply(img, mask, Fpixel);
	cv::namedWindow( "Display window", CV_WINDOW_AUTOSIZE );// Create a window for display.
	cv::imshow( "Display window", img );                   // Show our image inside it.
	cv::waitKey(0); 
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
	cv::namedWindow( "Display window", CV_WINDOW_AUTOSIZE );// Create a window for display.
	cv::imshow( "Display window", image );                   // Show our image inside it.
	cv::waitKey(0); 
	for(int i = 0; i < masks.size(); ++i)
	{
		auto t = getForMask(image, masks[i]);
		auto mid = std::get<0>(t);
		auto min = std::get<1>(t);
		auto max = std::get<2>(t);
		contrast(image, masks[i], mid, color[i], max, min, baseColor[i].prior);		
	}
	auto t = getForMask(image, outerMask);
	auto mid = std::get<0>(t);
	auto min = std::get<1>(t);
	auto max = std::get<2>(t);
	contrast(image, outerMask, mid, mid, max, min, default_prior, false);		
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
