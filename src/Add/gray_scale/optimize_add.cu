#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_SIZE 32

__global__ void kernel(const unsigned char *img1, const unsigned char *img2, unsigned char *result, int width, int height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ int shared_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int shared_B[BLOCK_SIZE][BLOCK_SIZE];

    shared_A[threadIdx.y][threadIdx.x] = (row < height && col < width) ? img1[row * width + col] : 0;
    shared_B[threadIdx.y][threadIdx.x] = (row < height && col < width) ? img2[row * width + col] : 0;

    __syncthreads();

    if (row < height && col < width)
    {
        int sum = shared_A[threadIdx.y][threadIdx.x] + shared_B[threadIdx.y][threadIdx.x];
        result[row * width + col] = (sum > 255) ? 255 : static_cast<unsigned char>(sum);
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

    cudaMemcpyAsync(d_img1, h_img1.data, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_img2, h_img2.data, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    kernel<<<grid, block>>>(d_img1, d_img2, d_result, width, height);

    unsigned char *h_result = new unsigned char[width * height];

    cudaMemcpyAsync(h_result, d_result, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cv::Mat result(height, width, CV_8U, h_result);

    cv::imshow("Result", result);
    cv::waitKey(0);

    cudaFree(d_img1);
    cudaFree(d_img2);
    cudaFree(d_result);
    delete[] h_result;

    return 0;
}