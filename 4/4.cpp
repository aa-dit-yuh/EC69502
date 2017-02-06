#include <opencv2/opencv.hpp>

#include <complex>
#include <experimental/filesystem>
#include <iostream>
#include <numeric>
#include <valarray>
#include <vector>

using namespace std;
using namespace std::complex_literals;
using Complex = std::complex<double>;

const auto N = 512;
const auto PI = std::acos(-1);
auto images = std::vector<cv::Mat>();
int imagePos, filterPos, freqPos;


namespace FFT {
    valarray<Complex> transform(const valarray<Complex> x)
    {
        auto n = x.size();
        auto X = valarray<Complex>(n);
        if (n == 1) return x;
        auto even = FFT::transform(x[slice(0, n/2, 2)]);
        auto odd  = FFT::transform(x[slice(1, n/2, 2)]);
        for (auto k = 0; k != n/2; k++) {
            auto twiddle = exp(-2 * PI * k / n * 1i);
            X[k] = even[k] + twiddle * odd[k];
            X[k + n/2] = even[k] - twiddle * odd[k];
        }
        return X;
    }

    valarray<Complex> inverseTransform(valarray<Complex> X)
    {
        X = X.apply([](Complex z) { return 1i * conj(z); });
        auto x = FFT::transform(X);
        x = x.apply([](Complex z) { return 1i * conj(z); });
        return x / static_cast<Complex>(x.size());
    }

    valarray<valarray<Complex>>& transpose(valarray<valarray<Complex>> &X)
    {
        for (auto i = 0; i != N; ++i) {
            for (auto j = 0; j != i; ++j) {
                swap(X[i][j], X[j][i]);
            }
        }
        return X;
    }

    valarray<valarray<Complex>> transform2d(const cv::Mat x)
    {
        auto X = valarray<valarray<Complex>>(valarray<Complex>(N), N);
        for (auto i = 0; i != N; ++i) {
            for (auto j = 0; j != N; ++j) {
                X[i][j] = static_cast<Complex>(x.at<uint8_t>(i, j));
            }
            X[i] = FFT::transform(X[i]);
        }
        X = FFT::transpose(X);
        for (auto i = 0; i != N; ++i) {
            X[i] = FFT::transform(X[i]);
        }
        return FFT::transpose(X);
    }

    valarray<valarray<Complex>> shift2d(const valarray<valarray<Complex>> &X)
    {
        auto ret = valarray<valarray<Complex>>(valarray<Complex>(N), N);
        for (auto i = 0; i != N; ++i) {
            for (auto j = 0; j != N; ++j) {
                ret[(N/2 + i)%N][(N/2 + j)%N] = log(1 + abs(X[i][j]));
            }
        }
        return ret;
    }

    cv::Mat inverseTransform2d(valarray<valarray<Complex>> X)
    {
        auto x = cv::Mat(N, N, CV_8UC1);
        for (auto i = 0; i != N; ++i) {
            for (auto j = 0; j != N; ++j) {
                X[i][j] = 1i * conj(X[i][j]);
            }
            X[i] = FFT::inverseTransform(X[i]);
        }
        X = FFT::transpose(X);
        for (auto i = 0; i != N; ++i) {
            X[i] = FFT::inverseTransform(X[i]);
        }
        X = FFT::transpose(X);
        for (auto i = 0; i != N; ++i) {
            for (auto j = 0; j != N; ++j) {
                x.at<uint8_t>(N-1-i, N-1-j) = static_cast<uint8_t>((1i * conj(X[i][j])).real());
            }
        }
        return x;
    }

    cv::Mat toMat(const valarray<valarray<Complex>> &X)
    {
        auto x = cv::Mat(N, N, CV_8UC1);
        for (auto i = 0; i != N; ++i) {
            for (auto j = 0; j != N; ++j) {
                x.at<uint8_t>(i, j) = 255/18 * static_cast<uint8_t>(X[i][j].real());
            }
        }
        return x;
    }
}

namespace Filter {
    namespace LowPass {
        enum _ {
            Ideal=0,
            Gaussian,
            Butterworth,
        };

        valarray<valarray<Complex>> ideal(const valarray<valarray<Complex>> X, int cutoff)
        {
            auto ret = valarray<valarray<Complex>>(valarray<Complex>(N), N);
            for (auto i = 0; i != N; ++i) {
                for (auto j = 0; j != N; ++j) {
                    if (abs(Complex((N/2 + i)%N - N/2, (N/2 + j)%N - N/2)) > cutoff) {
                        ret[i][j] = 0;
                    }
                    else {
                        ret[i][j] = X[i][j];
                    }
                }
            }
            return ret;
        }

        valarray<valarray<Complex>> gaussian(const valarray<valarray<Complex>> X, int stdDev)
        {
            auto ret = valarray<valarray<Complex>>(valarray<Complex>(N), N);
            for (auto i = 0; i != N; ++i) {
                for (auto j = 0; j != N; ++j) {
                    ret[i][j] = X[i][j] *
                        exp(-(pow((N/2 + i)%N - N/2, 2) + pow((N/2 + j)%N - N/2, 2))/(2*pow(stdDev, 2)));
                }
            }
            return ret;
        }

        valarray<valarray<Complex>> butterworth(const valarray<valarray<Complex>> X, int cutoff)
        {
            auto ret = valarray<valarray<Complex>>(valarray<Complex>(N), N);
            for (auto i = 0; i != N; ++i) {
                for (auto j = 0; j != N; ++j) {
                    ret[i][j] = X[i][j] /
                        (1 + pow(abs(Complex((N/2 + i)%N - N/2, (N/2 + j)%N - N/2))/cutoff, 4));
                }
            }
            return ret;
        }
    };
    namespace HighPass {
        enum _ {
            Ideal=3,
            Gaussian,
            Butterworth,
        };

        valarray<valarray<Complex>> ideal(const valarray<valarray<Complex>> X, int cutoff)
        {
            auto ret = valarray<valarray<Complex>>(valarray<Complex>(N), N);
            for (auto i = 0; i != N; ++i) {
                for (auto j = 0; j != N; ++j) {
                    if (abs(Complex((N/2 + i)%N - N/2, (N/2 + j)%N - N/2)) > cutoff) {
                        ret[i][j] = X[i][j];
                    }
                    else {
                        ret[i][j] = 0;
                    }
                }
            }
            return ret;
        }

        valarray<valarray<Complex>> gaussian(const valarray<valarray<Complex>> X, int stdDev)
        {
            auto ret = valarray<valarray<Complex>>(valarray<Complex>(N), N);
            for (auto i = 0; i != N; ++i) {
                for (auto j = 0; j != N; ++j) {
                    ret[i][j] = X[i][j] *
                        (1 - exp(-(pow((N/2 + i)%N - N/2, 2) + pow((N/2 + j)%N - N/2, 2))/(2*pow(stdDev, 2))));
                }
            }
            return ret;
        }

        valarray<valarray<Complex>> butterworth(const valarray<valarray<Complex>> X, int cutoff)
        {
            auto ret = valarray<valarray<Complex>>(valarray<Complex>(N), N);
            for (auto i = 0; i != N; ++i) {
                for (auto j = 0; j != N; ++j) {
                    ret[i][j] = X[i][j] * pow(abs(Complex((N/2 + i)%N - N/2, (N/2 + j)%N - N/2))/cutoff, 4) /
                        (1 + pow(abs(Complex((N/2 + i)%N - N/2, (N/2 + j)%N - N/2))/cutoff, 4));
                }
            }
            return ret;
        }
    };
};

static void callBack(int, void*)
{
    auto display = cv::Mat();
    auto input = images.at(imagePos);
    auto inputFFT = FFT::transform2d(input);

    auto output  = cv::Mat();
    auto outputFFT = valarray<valarray<Complex>>();

    switch(filterPos) {
    case Filter::LowPass::Ideal:
        outputFFT = Filter::LowPass::ideal(inputFFT, 20*(freqPos + 1));
        break;
    case Filter::HighPass::Ideal:
        outputFFT = Filter::HighPass::ideal(inputFFT, 20*(freqPos + 1));
        break;
    case Filter::LowPass::Gaussian:
        outputFFT = Filter::LowPass::gaussian(inputFFT, 20*(freqPos + 1));
        break;
    case Filter::HighPass::Gaussian:
        outputFFT = Filter::HighPass::gaussian(inputFFT, 20*(freqPos + 1));
        break;
    case Filter::LowPass::Butterworth:
        outputFFT = Filter::LowPass::butterworth(inputFFT, 10*(freqPos + 1));
        break;
    case Filter::HighPass::Butterworth:
        outputFFT = Filter::HighPass::butterworth(inputFFT, 10*(freqPos + 1));
        break;
    default:
        outputFFT = inputFFT;
    }
    cv::vconcat(
        input,
        FFT::toMat(FFT::shift2d(inputFFT)),
        input
    );
    cv::vconcat(
        FFT::inverseTransform2d(outputFFT),
        FFT::toMat(FFT::shift2d(outputFFT)),
        output
    );
    cv::hconcat(input, output, display);
    cv::imshow("Frequency Filtering", display);
}

int main()
{
    for (auto& file: std::experimental::filesystem::directory_iterator(".")) {
        auto imagePath = std::string(static_cast<std::string>(file.path()));
        if (imagePath.find(".jpg") != std::string::npos) {
            auto img = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
            if (img.rows == N && img.cols == N)
                images.push_back(img);
        }
    }

    cv::namedWindow("Frequency Filtering");
    cv::createTrackbar(
        "Image",
        "Frequency Filtering",
        &imagePos,
        images.size() - 1,
        callBack
    );
    cv::createTrackbar(
        "Filter",
        "Frequency Filtering",
        &filterPos,
        5,
        callBack
    );
    cv::createTrackbar(
        "Cutoff frequency",
        "Frequency Filtering",
        &freqPos,
        9,
        callBack
    );
    callBack(0, NULL);
    cv::waitKey(0);
    return 0;
}
