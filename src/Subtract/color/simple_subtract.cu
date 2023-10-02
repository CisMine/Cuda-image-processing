#include <opencv2/opencv.hpp>

__global__ void simple_subtract(const unsigned char *bgr1, const unsigned char *bgr2, unsigned char *result, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int idx = (y * width + x) * 3;
        int sumB = bgr1[idx] - bgr2[idx];
        int sumG = bgr1[idx + 1] - bgr2[idx + 1];
        int sumR = bgr1[idx + 2] - bgr2[idx + 2];

        result[idx] = (sumB < 0) ? 0 : static_cast<unsigned char>(sumB);
        result[idx + 1] = (sumG < 0) ? 0 : static_cast<unsigned char>(sumG);
        result[idx + 2] = (sumR < 0) ? 0 : static_cast<unsigned char>(sumR);
    }
}

int main(int argc, char *argv[])
{
    cv::Mat h_img1 = cv::imread("1.jpg");
    cv::Mat h_img2 = cv::imread("2.jpeg");

    // Resize images to the same size if needed
    cv::Size newSize(500, 500);
    cv::resize(h_img1, h_img1, newSize);
    cv::resize(h_img2, h_img2, newSize);

    int width = h_img1.cols;
    int height = h_img1.rows;

    unsigned char *d_img1, *d_img2, *d_result;
    cudaMalloc((void **)&d_img1, width * height * 3 * sizeof(unsigned char));
    cudaMalloc((void **)&d_img2, width * height * 3 * sizeof(unsigned char));
    cudaMalloc((void **)&d_result, width * height * 3 * sizeof(unsigned char));

    cudaMemcpy(d_img1, h_img1.data, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_img2, h_img2.data, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    simple_subtract<<<grid, block>>>(d_img1, d_img2, d_result, width, height);

    unsigned char *h_result = new unsigned char[width * height * 3];
    cudaMemcpy(h_result, d_result, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cv::Mat result(height, width, CV_8UC3, h_result);

    cv::imshow("Result", result);
    cv::waitKey(0);

    delete[] h_result;
    cudaFree(d_img1);
    cudaFree(d_img2);
    cudaFree(d_result);

    return 0;
}
