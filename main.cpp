#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <opencv2/opencv.hpp>


const int neighborhood_size = 3; //for masks
const double gauss_factor = 8;

const double a_max = 1.9; //limit contrast for small parts

//Dist
//OpenCV HSV: H [0-180], S [0-255], V [0-255]
const int gray_s = 19*255/100;
const int gray_v = 30*255/100;
const double inf_dist = 1000.;

struct BaseColor
{
	enum class Type {Colorful, Gray};

	int Hbegin;
	int Hend;
	Type type;
};

//OpenCV HSV: H [0-180], S [0-255], V [0-255]
std::vector<BaseColor> baseColor{ //before transformation
	{30, 70, BaseColor::Type::Colorful}, //Green
	{0, 0, BaseColor::Type::Gray},
	{20, 32, BaseColor::Type::Colorful} //Brown
};

std::vector<cv::Vec3b> color{ //we want to
	{12, 72, 62}, 
	{126, 112, 95},
	{17, 78, 62},
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
	if (a > a_max) //Limitation is needed because else there is too big color difference
	{
		std::cout << "A limitation\n";
		double rotation_point = (mid[0] + mid[1] + mid[2]) / 3;
		std::cout << "Rotation point: "<< rotation_point << "\n";
		double db = a*rotation_point - a_max*rotation_point;
		a = a_max;
		b += db;
		std::cout << "New A: " << a << ", b: " << b <<"\n";
	}
	auto Fpixel = [=](int x) { return a*x + b; };
	std::cout << "Applaying\n";
	//Applying
	for(int i = 0; i < img.rows; ++i) 
		for(int j = 0; j < img.cols; ++j)
			for(int ch = 0; ch < 3; ++ch)
			{
				int p = img.at<cv::Vec3b>(i, j)[ch];
				double k = mask.at<cv::Vec3b>(i, j)[0] / 255.;
				img.at<cv::Vec3b>(i, j)[ch] = std::max(std::min(
							Fpixel(p) * k + p * (1 - k)
							, 255.), 0.); 
			}
	
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
	//Comparator for set
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
	//Making masks
	std::cout << "Making masks\n";
	std::vector<cv::Mat> masks(freqColor.size());
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
			cv::GaussianBlur(mask, masks[i], cv::Size(neighborhood_size*gauss_factor*2+1, neighborhood_size*gauss_factor*2+1), 0);
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
	contrast(image, outerMask, mid, mid, max, min);		
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
