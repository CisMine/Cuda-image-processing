#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_SIZE 32

__global__ void kernel1(const unsigned char *img1, const unsigned char *img2, unsigned char *result, int width, int height)
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

cv::Mat add_gray(cv::Mat h_img1, cv::Mat h_img2)
{

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

    kernel1<<<grid, block>>>(d_img1, d_img2, d_result, width, height);

    unsigned char *h_result = new unsigned char[width * height];

    cudaMemcpy(h_result, d_result, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cv::Mat result(height, width, CV_8U, h_result);

    cudaFree(d_img1);
    cudaFree(d_img2);
    cudaFree(d_result);

    return result;
}


__global__ void kernel2(const uchar3 *bgr1, const uchar3 *bgr2, uchar3 *result, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ uchar3 smem1[32][32];
    __shared__ uchar3 smem2[32][32];

    if (x < width && y < height)
    {
        int idx = y * width + x;
        smem1[threadIdx.y][threadIdx.x] = bgr1[idx];
        smem2[threadIdx.y][threadIdx.x] = bgr2[idx];

        __syncthreads();

        int sumB = smem1[threadIdx.y][threadIdx.x].x + smem2[threadIdx.y][threadIdx.x].x;
        int sumG = smem1[threadIdx.y][threadIdx.x].y + smem2[threadIdx.y][threadIdx.x].y;
        int sumR = smem1[threadIdx.y][threadIdx.x].z + smem2[threadIdx.y][threadIdx.x].z;

        result[idx].x = sumB > 255 ? 255 : sumB;
        result[idx].y = sumG > 255 ? 255 : sumG;
        result[idx].z = sumR > 255 ? 255 : sumR;
    }
}

cv::Mat add_color(cv::Mat h_img1, cv::Mat h_img2)

{

    int width = h_img1.cols;
    int height = h_img1.rows;

    uchar3 *d_img1, *d_img2, *d_result;


    cudaMalloc((void **)&d_img1, width * height * sizeof(uchar3));
    cudaMalloc((void **)&d_img2, width * height * sizeof(uchar3));
    cudaMalloc((void **)&d_result, width * height * sizeof(uchar3));

    cudaMemcpy(d_img1, h_img1.data, width * height * sizeof(uchar3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_img2, h_img2.data, width * height * sizeof(uchar3), cudaMemcpyHostToDevice);

    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    kernel2<<<grid, block>>>(d_img1, d_img2, d_result, width, height);

    uchar3 *h_result = new uchar3[width * height];

    cudaMemcpy(h_result, d_result, width * height * sizeof(uchar3), cudaMemcpyDeviceToHost);

    cv::Mat result(height, width, CV_8UC3, h_result);

    cudaFree(d_img1);
    cudaFree(d_img2);
    cudaFree(d_result);

    return result;
}

__global__ void kernel3(const uchar3 *bgr1, const uchar3 *bgr2, uchar3 *result, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ uchar3 smem1[32][32];
    __shared__ uchar3 smem2[32][32];

    if (x < width && y < height)
    {
        int idx = y * width + x;
        smem1[threadIdx.y][threadIdx.x] = bgr1[idx];
        smem2[threadIdx.y][threadIdx.x] = bgr2[idx];

        __syncthreads();

        int sumB = smem1[threadIdx.y][threadIdx.x].x - smem2[threadIdx.y][threadIdx.x].x;
        int sumG = smem1[threadIdx.y][threadIdx.x].y - smem2[threadIdx.y][threadIdx.x].y;
        int sumR = smem1[threadIdx.y][threadIdx.x].z - smem2[threadIdx.y][threadIdx.x].z;

        result[idx].x = sumB < 0 ? 0 : sumB;
        result[idx].y = sumG < 0 ? 0 : sumG;
        result[idx].z = sumR < 0 ? 0 : sumR;
    }
}

cv::Mat subtract_color(cv::Mat h_img1, cv::Mat h_img2)

{

    int width = h_img1.cols;
    int height = h_img1.rows;

    uchar3 *d_img1, *d_img2, *d_result;

    cudaMalloc((void **)&d_img1, width * height * sizeof(uchar3));
    cudaMalloc((void **)&d_img2, width * height * sizeof(uchar3));
    cudaMalloc((void **)&d_result, width * height * sizeof(uchar3));

    cudaMemcpy(d_img1, h_img1.data, width * height * sizeof(uchar3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_img2, h_img2.data, width * height * sizeof(uchar3), cudaMemcpyHostToDevice);

    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    kernel3<<<grid, block>>>(d_img1, d_img2, d_result, width, height);

    uchar3 *h_result = new uchar3[width * height];

    cudaMemcpy(h_result, d_result, width * height * sizeof(uchar3), cudaMemcpyDeviceToHost);

    cv::Mat result(height, width, CV_8UC3, h_result);

    cudaFree(d_img1);
    cudaFree(d_img2);
    cudaFree(d_result);

    return result;
}

__global__ void kernel4(const unsigned char *img1, const unsigned char *img2, unsigned char *result, int width, int height)
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

cv::Mat subtract_gray(cv::Mat h_img1, cv::Mat h_img2)
{

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

    kernel4<<<grid, block>>>(d_img1, d_img2, d_result, width, height);

    unsigned char *h_result = new unsigned char[width * height];

    cudaMemcpy(h_result, d_result, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cv::Mat result(height, width, CV_8U, h_result);

    cudaFree(d_img1);
    cudaFree(d_img2);
    cudaFree(d_result);

    return result;
}
