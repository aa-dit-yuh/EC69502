#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <experimental/filesystem>
#include <iostream>
#include <vector>

namespace fs = std::experimental::filesystem;

std::vector<cv::Mat> unfiltered;
std::vector<cv::Mat> filtered;

int main()
{
    for (auto& file: fs::recursive_directory_iterator(".")) {
        std::string imagePath(static_cast<std::string>(file.path()));
        if (imagePath.find(".jpg") != std::string::npos) {
            unfiltered.push_back(cv::imread(imagePath));
        }
    }

    filtered = std::vector<cv::Mat>(unfiltered);
    cv::namedWindow("Spatial Filtering");
    cv::createTrackbar(
        "Image",
        "Spatial Filtering",
        NULL,
        filtered.size() - 1,
        [](int pos, void* data) {
            cv::Mat display;
            cv::hconcat(unfiltered.at(pos), filtered.at(pos), display);
            cv::imshow("Spatial Filtering", display);
        }
    );
    cv::createTrackbar(
        "Filter",
        "Spatial Filtering",
        NULL,
        filtered.size() - 1,
        [](int pos, void* data) {}
    );
    cv::createTrackbar(
        "Neighbourhood Size",
        "Spatial Filtering",
        NULL,
        filtered.size() - 1,
        [](int pos, void* data) {}
    );
    cv::waitKey(0);
    return 0;
}