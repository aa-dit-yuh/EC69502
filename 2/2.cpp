#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <cstdint>
#include <iostream>
#include <numeric>
#include <valarray>
#include <vector>
#include <string>

std::valarray<int> getHistogram(cv::Mat src)
{
    std::valarray<int> ret(256);
    for (auto i = src.begin<uint8_t>(); i != src.end<uint8_t>(); ++i)
        ret[*i]++;
    return ret;
}

std::valarray<int> getCumulativeHistogramNormalized(cv::Mat src)
{
    std::valarray<int> ret;
    if (src.channels() == 3) {
        cv::cvtColor(src, src, cv::COLOR_BGR2HSV);
        std::vector<cv::Mat> srcHSV;
        cv::split(src, srcHSV);
        ret = getHistogram(srcHSV[2]);
    }
    else
        ret = getHistogram(src);
    std::partial_sum(begin(ret), end(ret), begin(ret));
    ret = 255 * ret / ret.max();
    return ret;
}

void matchHistogram(cv::Mat& src, cv::Mat& dest, std::valarray<int>& refH_x)
{
    auto size = src.size();
    std::valarray<int> H_x = getCumulativeHistogramNormalized(src);
    for (auto i = 0; i != size.height; ++i)
        for (auto j = 0; j != size.width; ++j) {
            dest.at<uint8_t>(i, j) = *std::upper_bound(begin(refH_x), end(refH_x), H_x[src.at<uint8_t>(i, j)]);     // equalized(x,y) = H(image(x,y))
    }
}

int main()
{
    std::cout << "Enter choice:\n"
                 "1. Histogram Equalization\n"
                 "2. Histogram Matching\n"
              << std::endl;
    int choice; std::cin >> choice;

    std::cout << "Enter the filename of the input image: ";
    std::string inputFile; std::cin >> inputFile;
    cv::Mat input = cv::imread(inputFile, cv::IMREAD_UNCHANGED);
    if (!input.data) {
        std::cout << "Invalid input file\n" << std::endl;
        return main();
    }
    else
        cv::imshow("Original Image", input);

    std::valarray<int> refH_x(256);
    switch (choice) {
    case 1:
        std::iota(begin(refH_x), end(refH_x), 0);
        break;
    case 2: {
        std::cout << "Enter the filename of reference image: ";
        std::string refFile; std::cin >> refFile;
        cv::Mat ref = cv::imread(refFile, cv::IMREAD_UNCHANGED);
        if (!ref.data) {
            std::cout << "Invalid reference file\n" << std::endl;
            return main();
        }
        refH_x = getCumulativeHistogramNormalized(ref);
        break;
    }
    default:
        std::cout << "Invalid  choice\n" << std::endl;
        return main();
        break;
    }

    if (input.channels() == 3) {
        cv::cvtColor(input, input, cv::COLOR_BGR2HSV);
        std::vector<cv::Mat> inputHSV;
        cv::split(input, inputHSV);
        matchHistogram(inputHSV[2], inputHSV[2], refH_x);
        cv::merge(inputHSV, input);
        cv::cvtColor(input, input, cv::COLOR_HSV2BGR);
    }
    else
        matchHistogram(input, input, refH_x);
    cv::imshow("Histogram Equalized/Matched Image", input);
    cv::imwrite("matched-" + inputFile, input);
    cv::waitKey(0);

    return 0;
}
