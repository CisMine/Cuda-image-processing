#include "opencv2/opencv.hpp"
#include <opencv2/cudaarithm.hpp>

cv::Mat opencv_add(const cv::Mat &img1, const cv::Mat &img2)
{
    cv::cuda::GpuMat d_img1, d_img2, d_result;

    d_img1.upload(img1);
    d_img2.upload(img2);

    cv::cuda::add(d_img1, d_img2, d_result);

    cv::Mat result;
    d_result.download(result);

    return result;
}

int main()
{
    cv::Mat img1 = cv::imread("1.jpg");
    cv::Mat img2 = cv::imread("2.jpeg");

    // Resize images to the same size if needed
    // cv::Size newSize(, );
    // cv::resize(img1, img1, newSize);
    // cv::resize(img2, img2, newSize);

    cv::Mat result = opencv_add(img1, img2);

    cv::imshow("Result", result);
    cv::waitKey();

    return 0;
}
