#include <opencv2/opencv.hpp>

#include <array>
#include <iostream>
#include <numeric>
#include <valarray>

using Kernel = std::valarray<std::valarray<int>>;

const uint8_t BIN_THRESH = 135;
cv::Mat inputImage, inputBinary;
int operationPos = 0, kernelPos = 0;

namespace cv {
    static cv::Mat imcvtBinary(cv::Mat &input)
    {
        cv::Mat ret(input.size(), CV_8UC1);
        for (int i = 0; i < input.rows; i++) {
            for (int j = 0; j < input.cols; j++) {
                if (input.at<uint8_t>(i, j) >= BIN_THRESH) {
                    ret.at<uint8_t>(i, j) = static_cast<uint8_t>(255);
                }
                else {
                    ret.at<uint8_t>(i, j) = static_cast<uint8_t>(0);
                }
            }
        }
        return ret;
    }
};

enum KERNEL {
    RECT_1x2 = 0,
    DIAMOND_3x3,
    SQUARE_3x3,
    SQUARE_9x9,
    SQUARE_15x15,
};

static std::array<Kernel, 5> kernels = {
    Kernel({
        { 0, 1, 1 },
    }),
    Kernel({
        { 0, 1, 0 },
        { 1, 1, 1 },
        { 0, 1, 0 },
    }),
    Kernel({
        { 1, 1, 1 },
        { 1, 1, 1 },
        { 1, 1, 1 },
    }),
    Kernel({
        { 1, 1, 1, 1, 1, 1, 1, 1, 1 },
        { 1, 1, 1, 1, 1, 1, 1, 1, 1 },
        { 1, 1, 1, 1, 1, 1, 1, 1, 1 },
        { 1, 1, 1, 1, 1, 1, 1, 1, 1 },
        { 1, 1, 1, 1, 1, 1, 1, 1, 1 },
        { 1, 1, 1, 1, 1, 1, 1, 1, 1 },
        { 1, 1, 1, 1, 1, 1, 1, 1, 1 },
        { 1, 1, 1, 1, 1, 1, 1, 1, 1 },
        { 1, 1, 1, 1, 1, 1, 1, 1, 1 },
    }),
    Kernel({
        { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
        { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
        { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
        { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
        { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
        { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
        { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
        { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
        { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
        { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
        { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
        { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
        { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
        { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
        { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
    }),
};

namespace morphology {
    enum _ {
        ERODE = 0,
        DILATE,
        OPEN,
        CLOSE,
    };

    static cv::Mat erode(const cv::Mat &input, const Kernel &h)
    {
        cv::Mat ret = input.clone();
        for (int i = 0; i != input.rows; ++i) {
            for (int j = 0; j != input.cols; ++j) {
                for (int k = 0; k != h.size(); ++k) {
                    for (int l = 0; l != h[0].size(); ++l) {
                        int row = i + k - h.size()/2;
                        int col = j + l - h[0].size()/2;
                        if (h[k][l] && 0 <= row < input.rows && 0 <= col < input.cols) {
                            if (!input.at<uint8_t>(row, col))
                                ret.at<uint8_t>(i, j) = 0;
                        }
                    }
                }
            }
        }
        return ret;
    }

    static cv::Mat dilate(const cv::Mat &input, const Kernel &h)
    {
        cv::Mat ret = input.clone();
        for (int i = 0; i != input.rows; ++i) {
            for (int j = 0; j != input.cols; ++j) {
                for (int k = 0; k != h.size(); ++k) {
                    for (int l = 0; l != h[0].size(); ++l) {
                        int row = i + k - h.size()/2;
                        int col = j + l - h[0].size()/2;
                        if (h[k][l] && 0 <= row < input.rows && 0 <= col < input.cols) {
                            if (input.at<uint8_t>(row, col))
                                ret.at<uint8_t>(i, j) = 255;
                        }
                    }
                }
            }
        }
        return ret;
    }

    static cv::Mat open(const cv::Mat &input, const Kernel &h)
    {
        cv::Mat temp = erode(input, h);
        return dilate(temp, h);
    }

    static cv::Mat close(const cv::Mat &input, const Kernel &h)
    {
        cv::Mat temp = dilate(input, h);
        return erode(temp, h);
    }
};

static void callBack(int, void*)
{
    cv::Mat display, output;
    switch (operationPos) {
    case morphology::ERODE:
        output = morphology::erode(inputBinary, kernels[kernelPos]); 
        break;
    case morphology::DILATE:
        output = morphology::dilate(inputBinary,kernels[kernelPos]);
        break;
    case morphology::OPEN:
        output = morphology::open(inputBinary, kernels[kernelPos]);  
        break;
    case morphology::CLOSE:
        output = morphology::close(inputBinary, kernels[kernelPos]);
        break;
    }
    cv::hconcat(inputImage, output, display);
    cv::imshow("Morphological Operations", display);
}

int main()
{
    std::cout << "Enter input filename: ";
    std::string inputFile;
    std::cin >> inputFile;

    inputImage  = cv::imread(inputFile, cv::IMREAD_GRAYSCALE);
    inputBinary = cv::imcvtBinary(inputImage);

    cv::namedWindow("Morphological Operations");
    cv::createTrackbar(
        "Operation",
        "Morphological Operations",
        &operationPos,
        3,
        callBack
    );
    cv::createTrackbar(
        "Structuring Element",
        "Morphological Operations",
        &kernelPos,
        4,
        callBack
    );
    callBack(0, NULL);
    cv::waitKey(0);

    return 0;
}
