#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <experimental/filesystem>
#include <iostream>
#include <numeric>
#include <vector>

namespace fs = std::experimental::filesystem;

enum FILTER {
    MEAN = 0,
    MEDIAN,
    GRADIENT_HORIZONTAL,
    GRADIENT_VERTICAL,
    LAPLACIAN,
    SOBEL_HORIZONTAL,
    SOBEL_VERTICAL,
    SOBEL_DIAGONAL,
};
enum KERNEL {
    THREE = 0,
    FIVE,
    SEVEN,
};

std::vector<cv::Mat> unfiltered;
int imagePos = 0, filterPos = 0, kernelPos = 0;

static void Mean(const cv::Mat &input, int kernel)
{
    cv::Mat display, output(input.size(), input.type());
    std::vector<uint8_t> neighbourhood;
    for (auto i = 0; i != output.rows; ++i) {
        for (auto j = 0; j != output.cols; ++j) {
            neighbourhood.clear();
            for (auto k = -kernel/2; k <= kernel/2; ++k) {
                for (auto l = -kernel/2; l <= kernel/2; ++l) {
                    if ((i+k >=0 && i+k < output.rows) && (j+l >=0 && j+l < output.cols)) {
                        neighbourhood.push_back(input.at<uint8_t>(i+k, j+l));
                    }
                }
            }
            output.at<uint8_t>(i, j) = std::accumulate(neighbourhood.begin(), neighbourhood.end(), 0)/neighbourhood.size();
        }
    }
    cv::hconcat(input, output, display);
    cv::imshow("Spatial Filtering", display);
}

static void Median(const cv::Mat &input, int kernel)
{
    cv::Mat display, output(input.size(), input.type());
    std::vector<uint8_t> neighbourhood;
    for (auto i = 0; i != output.rows; ++i) {
        for (auto j = 0; j != output.cols; ++j) {
            neighbourhood.clear();
            for (auto k = -kernel/2; k <= kernel/2; ++k) {
                for (auto l = -kernel/2; l <= kernel/2; ++l) {
                    if ((i+k >=0 && i+k < output.rows) && (j+l >=0 && j+l < output.cols)) {
                        neighbourhood.push_back(input.at<uint8_t>(i+k, j+l));
                    }
                }
            }
            auto size = neighbourhood.size();
            std::sort(neighbourhood.begin(), neighbourhood.end());
            if (size % 2 == 0) {
                output.at<uint8_t>(i, j) = (neighbourhood.at(size/2 - 1) + neighbourhood.at(size/2 + 1))/2;
            }
            else {
                output.at<uint8_t>(i, j) = neighbourhood.at(size/2);
            }
        }
    }
    cv::hconcat(input, output, display);
    cv::imshow("Spatial Filtering", display);
}

static void callBack(int, void*)
{
    int kernel;
    cv::Mat input = unfiltered.at(imagePos);
    switch (kernelPos) {
        case KERNEL::THREE: kernel = 3; break;
        case KERNEL::FIVE : kernel = 5; break;
        case KERNEL::SEVEN: kernel = 7; break;
    }
    switch(filterPos) {
    case FILTER::MEAN:
        Mean(input, kernel);
        break;
    case FILTER::MEDIAN:
        Median(input, kernel);
        break;
    }
}

int main()
{
    for (auto& file: fs::recursive_directory_iterator(".")) {
        std::string imagePath(static_cast<std::string>(file.path()));
        if (imagePath.find(".jpg") != std::string::npos) {
            unfiltered.push_back(cv::imread(imagePath));
        }
    }

    cv::namedWindow("Spatial Filtering");
    cv::createTrackbar(
        "Image",
        "Spatial Filtering",
        &imagePos,
        unfiltered.size() - 1,
        callBack
    );
    cv::createTrackbar(
        "Filter",
        "Spatial Filtering",
        &filterPos,
        7,
        callBack
    );
    cv::createTrackbar(
        "Kernel Size",
        "Spatial Filtering",
        &kernelPos,
        2,
        callBack
    );
    callBack(0, NULL);
    cv::waitKey(0);
    return 0;
}
