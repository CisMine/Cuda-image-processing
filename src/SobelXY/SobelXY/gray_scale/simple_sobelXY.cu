#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#define MASK_DIM 3
#define MASK_OFFSET 1

__constant__ float mask[MASK_DIM * MASK_DIM] = {
    1.0f, 0.0f, -1.0f,
    2.0f, 0.0f, -2.0f,
    1.0f, 0.0f, -1.0f};

__constant__ float mask2[MASK_DIM * MASK_DIM] = {
    1.0f, 2.0f, 1.0f,
    0.0f, 0.0f, 0.0f,
    -1.0f, -2.0f, -1.0f};

__global__ void simple_sobelXY(const unsigned char *srcImage, unsigned char *result, int width, int height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if ((col >= MASK_OFFSET) && (col < (width - MASK_OFFSET)) && (row >= MASK_OFFSET) && (row < (height - MASK_OFFSET)))
    {
        float Gx = 0.0f;
        float Gy = 0.0f;

        for (int ky = -MASK_OFFSET; ky <= MASK_OFFSET; ky++)
        {
            for (int kx = -MASK_OFFSET; kx <= MASK_OFFSET; kx++)
            {
                float fl = static_cast<float>(srcImage[(row + ky) * width + (col + kx)]);
                Gx += fl * mask[(ky + MASK_OFFSET) * MASK_DIM + (kx + MASK_OFFSET)];
                Gy += fl * mask2[(ky + MASK_OFFSET) * MASK_DIM + (kx + MASK_OFFSET)];
            }
        }

        float Gxy_abs = sqrtf(Gx * Gx + Gy * Gy);
        Gxy_abs = (Gxy_abs > 255) ? 255 : Gxy_abs;

        result[row * width + col] = static_cast<unsigned char>(Gxy_abs);
    }
}

int main()
{
    cv::Mat image = cv::imread("cameraman.png", cv::IMREAD_GRAYSCALE);

    int width = image.cols;
    int height = image.rows;

    unsigned char *d_image, *d_result;

    cudaMalloc((void **)&d_image, width * height * sizeof(unsigned char));
    cudaMalloc((void **)&d_result, width * height * sizeof(unsigned char));

    cudaMemcpy(d_image, image.data, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 threads(32, 32);
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

    simple_sobelXY<<<blocks, threads>>>(d_image, d_result, width, height);

    unsigned char *h_result = new unsigned char[width * height];

    cudaMemcpy(h_result, d_result, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cv::Mat result(height, width, CV_8U, h_result);

    cudaFree(d_image);
    cudaFree(d_result);

    cv::imshow("Result", result);
    cv::waitKey(0);

    delete[] h_result;

    return 0;
}
