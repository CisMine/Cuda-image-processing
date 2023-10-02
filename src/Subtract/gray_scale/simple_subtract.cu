#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

__global__ void simple_subtract(const unsigned char *img1, const unsigned char *img2, unsigned char *result, int width, int height)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int idx = y * width + x;
        int sum = img1[idx] - img2[idx];
        result[idx] = (sum < 0) ? 0 : static_cast<unsigned char>(sum);
    }
}

int main(int argc, char *argv[])
{

    cv::Mat h_img1 = cv::imread("circles.png", 0);
    cv::Mat h_img2 = cv::imread("cameraman.png", 0);

    // Resize images to the same size if needed
    // cv::Size newSize(, );
    // cv::resize(h_img1, h_img1, newSize);
    // cv::resize(h_img2, h_img2, newSize);

    int width = h_img1.cols;
    int height = h_img1.rows;

    unsigned char *d_img1, *d_img2, *d_result;

    cudaMalloc((void **)&d_img1, width * height * sizeof(unsigned char));
    cudaMalloc((void **)&d_img2, width * height * sizeof(unsigned char));
    cudaMalloc((void **)&d_result, width * height * sizeof(unsigned char));

    cudaMemcpy(d_img1, h_img1.data, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_img2, h_img2.data, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    simple_subtract<<<grid, block>>>(d_img2, d_img1, d_result, width, height);

    unsigned char *h_result = new unsigned char[width * height];

    cudaMemcpy(h_result, d_result, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cv::Mat result(height, width, CV_8U, h_result);

    cv::imshow("Result", result);
    cv::waitKey(0);

    cudaFree(d_img1);
    cudaFree(d_img2);
    cudaFree(d_result);
    delete[] h_result;

    return 0;
}