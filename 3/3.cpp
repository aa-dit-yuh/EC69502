#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <experimental/filesystem>
#include <iostream>
#include <numeric>
#include <vector>

namespace fs = std::experimental::filesystem;
typedef const std::vector<std::vector<int>> Kernel;

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

Kernel MEAN_3 = {
    { 1,  1,  1 },
    { 1,  1,  1 },
    { 1,  1,  1 },
};

Kernel MEAN_5 = {
    { 1,  1,  1,  1,  1 },
    { 1,  1,  1,  1,  1 },
    { 1,  1,  1,  1,  1 },
    { 1,  1,  1,  1,  1 },
    { 1,  1,  1,  1,  1 },
};

Kernel MEAN_7 = {
    { 1,  1,  1,  1,  1,  1,  1 },
    { 1,  1,  1,  1,  1,  1,  1 },
    { 1,  1,  1,  1,  1,  1,  1 },
    { 1,  1,  1,  1,  1,  1,  1 },
    { 1,  1,  1,  1,  1,  1,  1 },
    { 1,  1,  1,  1,  1,  1,  1 },
    { 1,  1,  1,  1,  1,  1,  1 },
};

Kernel GRADIENT_H_3 =   {
    {-1, -1, -1 },
    { 0,  0,  0 },
    { 1,  1,  1 },
};

Kernel GRADIENT_H_5 = {
    { -1, -1, -1, -1, -1 },
    { -2, -2, -2, -2, -2 },
    {  0,  0,  0,  0,  0 },
    {  2,  2,  2,  2,  2 },
    {  1,  1,  1,  1,  1 },
};

Kernel GRADIENT_H_7 =  {
    { -1, -1, -1, -1, -1, -1, -1 },
    { -2, -2, -2, -2, -2, -2, -2 },
    { -3, -3, -3, -3, -3, -3, -3 },
    {  0,  0,  0,  0,  0,  0,  0 },
    {  3,  3,  3,  3,  3,  3,  3 },
    {  2,  2,  2,  2,  2,  2,  2 },
    {  1,  1,  1,  1,  1,  1,  1 },
};

Kernel GRADIENT_V_3 = {
    { 1, 0, -1 },
    { 1, 0, -1 },
    { 1, 0, -1 },
};

Kernel GRADIENT_V_5 ={
    { -1, -2,  0,  2,  1},
    { -1, -2,  0,  2,  1},
    { -1, -2,  0,  2,  1},
    { -1, -2,  0,  2,  1},
    { -1, -2,  0,  2,  1}
};

Kernel GRADIENT_V_7 = {
    { -1, -2, -3,  0,  3,  2,  1},
    { -1, -2, -3,  0,  3,  2,  1},
    { -1, -2, -3,  0,  3,  2,  1},
    { -1, -2, -3,  0,  3,  2,  1},
    { -1, -2, -3,  0,  3,  2,  1},
    { -1, -2, -3,  0,  3,  2,  1},
    { -1, -2, -3,  0,  3,  2,  1},
};

Kernel LAPLACIAN_3 = {
    { -1, -1, -1},
    { -1,  8, -1},
    { -1, -1, -1},
};

Kernel LAPLACIAN_5 = {
    { -1,  3, -4, -3, -1},
    { -3,  0,  6,  0, -3},
    { -4,  6, 20,  6, -4},
    { -3,  0,  6,  0, -3},
    { -1, -3, -4, -3, -1},
};

Kernel LAPLACIAN_7 = {
    { -2, -3, -4, -6, -4, -3, -2},
    { -3, -5, -4, -3, -4, -5, -3},
    { -4, -4,  9, 20,  9, -4, -4},
    { -6, -3, 20, 36, 20, -3, -6},
    { -4, -4,  9, 20,  9, -4, -4},
    { -3, -5, -4, -3, -4, -5, -3},
    { -2, -3, -4, -6, -4, -3, -2},
};

Kernel SOBEL_H_3 = {
    {  1,  2,  1},
    {  0,  0,  0},
    { -1, -2, -1},
};

Kernel SOBEL_H_5 = {
    {  1,   4,   7,   4,  1},
    {  2,  10,  17,  10,  2},
    {  0,   0,   0,   0,  0},
    { -2, -10, -17, -10, -2},
    { -1,  -4,  -7,  -4, -1},
};

Kernel SOBEL_H_7 = {
    {  1,   4,   9,  13,   9,   4,  1},
    {  3,  11,  26,  34,  26,  11,  3},
    {  3,  13,  30,  40,  30,  13,  3},
    {  0,   0,   0,   0,   0,   0,  0},
    { -3, -13, -30, -40, -30, -13, -3},
    { -3, -11, -26,  34, -26, -11, -3},
    { -1,  -4,  -9, -13,  -9,  -4, -1},
};

Kernel SOBEL_V_3 = {
    { -1,  0,  1 },
    { -2,  0,  2 },
    { -1,  0,  1 },
};

Kernel SOBEL_V_5 = {
    { -1,  -2, 0,  2, 1},
    { -4, -10, 0, 10, 4},
    { -7, -17, 0, 17, 7},
    { -4, -10, 0, 10, 4},
    { -1,  -2, 0, -2, 1},
};

Kernel SOBEL_V_7 = {
    {  -1,  -3,  -3, 0, 3,  3,   1},
    {  -4, -11, -13, 0, 13, 11,  4},
    {  -9, -26, -30, 0, 30, 26,  9},
    { -13, -34, -40, 0, 40, 34, 13},
    {  -9, -26, -30, 0, 30, 26,  9},
    {  -4, -11, -13, 0, 13, 11,  4},
    {   1,  -3,  -3, 0,  3,  3,  1},
};

Kernel SOBEL_D_3 = {
    {  0,  1,  2},
    { -1,  0,  1},
    { -2, -1,  0},
};

Kernel SOBEL_D_5 = {
    {  0,  1,  2,  3,  4},
    { -1,  0,  1,  2,  3},
    { -2, -1,  0,  1,  2},
    { -3, -2, -1,  0,  1},
    { -4, -3, -2, -1,  0},
};

Kernel SOBEL_D_7 = {
    {  0,  1,  2,  3,  4,  5,  6},
    { -1,  0,  1,  2,  3,  4,  5},
    { -2, -1,  0,  1,  2,  3,  4},
    { -3, -2, -1,  0,  1,  2,  3},
    { -4, -3, -2, -1,  0,  1,  2},
    { -5, -4, -3, -2, -1,  0,  1},
    { -6, -5, -4, -3, -2, -1,  0},
};


static void convolute(const cv::Mat &input, cv::Mat &output, Kernel h)
{
    for (auto i = 0; i != output.rows; ++i) {
        for (auto j = 0; j != output.cols; ++j) {
            auto sum = 0, den=0;
            for(auto k = 0; k != h.size(); ++k) {
                for(auto l = 0; l != h.size(); ++l) {
                    sum += input.at<uint8_t>(i+k - h.size()/2, j+l - h.size()/2) * h[k][l];
                    if (h[k][l] >= 0)                                               // Normalize the filter output
                        den += h[k][l];                                             // using positive sum of the kernel
                }
            }
            output.at<uint8_t>(i, j) = abs(sum/den);                                // Use absolute value to flip negative gradients
        }
    }
}

static void Median(const cv::Mat &input, cv::Mat &output, int kernel)
{
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
}

static void callBack(int, void*)
{
    std::string filterText, kernel;
    cv::Mat display, input = unfiltered.at(imagePos), output(input.size(), input.type());
    switch(filterPos) {
    case FILTER::MEAN:
        filterText = "Mean: ";
        switch (kernelPos) {
            case KERNEL::THREE: kernel = "3"; convolute(input, output, MEAN_3); break;
            case KERNEL::FIVE : kernel = "5"; convolute(input, output, MEAN_5); break;
            case KERNEL::SEVEN: kernel = "7"; convolute(input, output, MEAN_7); break;
        }
        break;
    case FILTER::MEDIAN:
        filterText = "Median: ";
        switch (kernelPos) {
            case KERNEL::THREE: kernel = "3"; Median(input, output, 3); break;
            case KERNEL::FIVE : kernel = "5"; Median(input, output, 5); break;
            case KERNEL::SEVEN: kernel = "7"; Median(input, output, 7); break;
        }
        break;
    case FILTER::GRADIENT_HORIZONTAL:
        filterText = "Gradient Horizontal: ";
        switch (kernelPos) {
            case KERNEL::THREE: kernel = "3"; convolute(input, output, GRADIENT_H_3); break;
            case KERNEL::FIVE : kernel = "5"; convolute(input, output, GRADIENT_H_5); break;
            case KERNEL::SEVEN: kernel = "7"; convolute(input, output, GRADIENT_H_7); break;
        }
        break;
    case FILTER::GRADIENT_VERTICAL:
        filterText = "Gradient Vertical: ";
        switch (kernelPos) {
            case KERNEL::THREE: kernel = "3"; convolute(input, output, GRADIENT_V_3); break;
            case KERNEL::FIVE : kernel = "5"; convolute(input, output, GRADIENT_V_5); break;
            case KERNEL::SEVEN: kernel = "7"; convolute(input, output, GRADIENT_V_7); break;
        }
        break;
    case FILTER::LAPLACIAN:
        filterText = "Laplacian: ";
        switch (kernelPos) {
            case KERNEL::THREE: kernel = "3"; convolute(input, output, LAPLACIAN_3); break;
            case KERNEL::FIVE : kernel = "5"; convolute(input, output, LAPLACIAN_5); break;
            case KERNEL::SEVEN: kernel = "7"; convolute(input, output, LAPLACIAN_7); break;
        }
        break;
    case FILTER::SOBEL_HORIZONTAL:
        filterText = "Sobel Horizontal: ";
        switch (kernelPos) {
            case KERNEL::THREE: kernel = "3"; convolute(input, output, SOBEL_H_3); break;
            case KERNEL::FIVE : kernel = "5"; convolute(input, output, SOBEL_H_5); break;
            case KERNEL::SEVEN: kernel = "7"; convolute(input, output, SOBEL_H_7); break;
        }
        break;
    case FILTER::SOBEL_VERTICAL:
        filterText = "Sobel Vertical: ";
        switch (kernelPos) {
            case KERNEL::THREE: kernel = "3"; convolute(input, output, SOBEL_V_3); break;
            case KERNEL::FIVE : kernel = "5"; convolute(input, output, SOBEL_V_5); break;
            case KERNEL::SEVEN: kernel = "7"; convolute(input, output, SOBEL_V_7); break;
        }
        break;
    case FILTER::SOBEL_DIAGONAL:
        filterText = "Sobel Diagonal: ";
        switch (kernelPos) {
            case KERNEL::THREE: kernel = "3"; convolute(input, output, SOBEL_D_3); break;
            case KERNEL::FIVE : kernel = "5"; convolute(input, output, SOBEL_D_5); break;
            case KERNEL::SEVEN: kernel = "7"; convolute(input, output, SOBEL_D_7); break;
        }
        break;
    }
    cv::hconcat(input, output, display);
    std::string text(filterText + "Kernel=" + kernel);
    cv::putText(display, text, cv::Point(768, 20), cv::FONT_HERSHEY_PLAIN, 1, 0);
    cv::putText(display, text, cv::Point(768, 50), cv::FONT_HERSHEY_PLAIN, 1, 255);
    cv::imshow("Spatial Filtering", display);
}

int main()
{
    for (auto& file: fs::recursive_directory_iterator(".")) {
        std::string imagePath(static_cast<std::string>(file.path()));
        if (imagePath.find(".jpg") != std::string::npos) {
            unfiltered.push_back(cv::imread(imagePath, cv::IMREAD_GRAYSCALE));
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
