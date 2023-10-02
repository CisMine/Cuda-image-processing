#include "my_lib.h"

int main()
{
    cv::Mat img = cv::imread("input");
    cv::Mat img2 = cv::imread("input");

    cv::Mat result1 = add_gray(img, img2);
    cv::Mat result2 = add_color(img, img2);

    return 0;
}
