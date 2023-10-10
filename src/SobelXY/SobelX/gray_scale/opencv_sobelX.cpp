#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <iostream>

int main()
{
    cv::Mat img = cv::imread("cameraman.png", cv::IMREAD_GRAYSCALE);

    cv::cuda::GpuMat d_img;
    d_img.upload(img);

    cv::Ptr<cv::cuda::Filter> filter3x3 = cv::cuda::createSobelFilter(CV_8U, CV_8U, 1, 0);
    cv::cuda::GpuMat d_result;
    filter3x3->apply(d_img, d_result);

    cv::Mat result;
    d_result.download(result);

    cv::imshow("result_cv", result);
    cv::waitKey();

    return 0;
}
