#include <iostream>
#include <opencv2/cudafilters.hpp>
#include "opencv2/opencv.hpp"
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>

int main()
{
    cv::Mat img = cv::imread("cameraman.png", cv::IMREAD_GRAYSCALE);

    cv::cuda::GpuMat d_img;
    d_img.upload(img);

    cv::cuda::GpuMat d_resultx, d_resulty, d_resultxy;

    cv::Ptr<cv::cuda::Filter> filterx, filtery;
    filterx = cv::cuda::createSobelFilter(CV_8UC1, CV_8UC1, 1, 0);
    filtery = cv::cuda::createSobelFilter(CV_8UC1, CV_8UC1, 0, 1);

    filterx->apply(d_img, d_resultx);
    filtery->apply(d_img, d_resulty);

    cv::cuda::add(d_resultx, d_resulty, d_resultxy);

    cv::Mat result;
    d_resultxy.download(result);

    cv::imshow("result_cv", result);
    cv::waitKey();

    return 0;
}
