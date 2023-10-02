#include <opencv2/opencv.hpp>

__global__ void simple_add(const uchar3 *bgr1, const uchar3 *bgr2, uchar3 *result, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int idx = (y * width + x);
        int sumB = bgr1[idx].x + bgr2[idx].x;
        int sumG = bgr1[idx].y + bgr2[idx].y;
        int sumR = bgr1[idx].z + bgr2[idx].z;

        result[idx].x = (sumB > 255) ? 255 : sumB;
        result[idx].y = (sumG > 255) ? 255 : sumG;
        result[idx].z = (sumR > 255) ? 255 : sumR;
    }
}

int main(int argc, char *argv[])
{
    cv::Mat h_img1 = cv::imread("1.jpg");
    cv::Mat h_img2 = cv::imread("2.jpeg");

    // Resize images to the same size if needed
    cv::Size newSize(1003, 1005);
    cv::resize(h_img1, h_img1, newSize);
    cv::resize(h_img2, h_img2, newSize);

    int width = h_img1.cols;
    int height = h_img1.rows;

    uchar3 *d_img1, *d_img2, *d_result;

    // Use cudaMalloc, not cudaMallocHost for device memory allocation
    cudaMalloc((void **)&d_img1, width * height * sizeof(uchar3));
    cudaMalloc((void **)&d_img2, width * height * sizeof(uchar3));
    cudaMalloc((void **)&d_result, width * height * sizeof(uchar3));

    cudaMemcpy(d_img1, h_img1.data, width * height * sizeof(uchar3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_img2, h_img2.data, width * height * sizeof(uchar3), cudaMemcpyHostToDevice);

    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    simple_add<<<grid, block>>>(d_img1, d_img2, d_result, width, height);

    uchar3 *h_result = new uchar3[width * height];

    cudaMemcpy(h_result, d_result, width * height * sizeof(uchar3), cudaMemcpyDeviceToHost);

    cv::Mat result(height, width, CV_8UC3, h_result);

    cv::imshow("Result", result);
    cv::waitKey(0);

    delete[] h_result;
    cudaFree(d_img1);
    cudaFree(d_img2);
    cudaFree(d_result);

    return 0;
}
