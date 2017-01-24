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
    for (auto i = 0; i != size.height; ++i) {
        for (auto j = 0; j != size.width; ++j) {
            for (int k = 0; k != 256; ++k) {
                if (refH_x[k] >= H_x[src.at<uint8_t>(i, j)]) {
                    dest.at<uint8_t>(i, j) = k;
                    break;
                }
            }
        }
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
    else {
        cv::imshow("Original Image", input);
        cv::Mat inputHist(512, 512, CV_8UC1, 255);
        auto hist = getHistogram(input);
        for (int i = 0; i != 256; i++)
            cv::line(inputHist, cv::Point(2*i, 512), cv::Point(2*i, 512 - (512*hist[i]/hist.max())), 0);
        cv::imshow("Input Histogram", inputHist);
        cv::imwrite("histogram-" + inputFile, inputHist);
    }

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
        cv::Mat refHist(512, 512, CV_8UC1, 255);
        auto hist = getHistogram(ref);
        for (int i = 0; i != 256; i++)
            cv::line(refHist, cv::Point(2*i, 512), cv::Point(2*i, 512 - (512*hist[i]/hist.max())), 0);
        cv::imshow("Reference Histogram", refHist);
        cv::imwrite("histogram-ref-" + inputFile, refHist);
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

    cv::Mat outputHist(512, 512, CV_8UC1, 255);
    auto hist = getHistogram(input);
    for (int i = 0; i != 256; i++)
        cv::line(outputHist, cv::Point(2*i, 512), cv::Point(2*i, 512 - (512*hist[i]/hist.max())), 0);
    cv::imshow("Output Histogram", outputHist);
    cv::imwrite("histogram-out-" + inputFile, outputHist);

    cv::waitKey(0);

    return 0;
}
