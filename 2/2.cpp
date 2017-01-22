#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/plot.hpp>

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

void equalizeHistogram(cv::Mat& src, cv::Mat& dest)
{
    auto size = src.size();
    std::valarray<int> h_x = getHistogram(src), H_x(256);
    std::partial_sum(begin(h_x), end(h_x), begin(H_x));                     // Calculate H(x) = integ{h(x)}
    H_x = 255 * H_x / H_x.max();                                            // Normalize H(x)
    for (auto i = 0; i != size.height; ++i)
        for (auto j = 0; j != size.width; ++j)
            dest.at<uint8_t>(i, j) = H_x[src.at<uint8_t>(i, j)];     // equalized(x,y) = H(image(x,y))
}

int main()
{
    std::cout << "Enter choice:\n"
                 "1. Histogram Equalization\n"
                 "2. Histogram Matching\n"
              << std::endl;
    int choice; std::cin >> choice;

    switch (choice) {
    case 1: {
        std::cout << "Enter the filename of the image: ";
        std::string file; std::cin >> file;

        cv::Mat image = cv::imread(file, cv::IMREAD_UNCHANGED);
        if (!image.data) {
            std::cout << "Invalid file\n" << std::endl;
            return main();
        }
        cv::imshow("Original Image", image);

        if (image.channels() == 3) {
            cv::cvtColor(image, image, cv::COLOR_BGR2HSV);
            std::vector<cv::Mat> HSV;
            cv::split(image, HSV);
            equalizeHistogram(HSV[0], HSV[0]);
            cv::cvtColor(image, image, cv::COLOR_HSV2BGR);
            cv::imshow("Histogram Equalized Image", image);
            cv::waitKey(0);
        }
        else {
            cv::Mat equalized(image);
            equalizeHistogram(image, equalized);
            cv::imshow("Histogram Equalized", equalized);
            cv::waitKey(0);
        }
        break;
    }
    default:
        std::cout << "Invalid  choice\n" << std::endl;
        return main();
        break;
    }
    return 0;
}
