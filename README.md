

<p align="center">
 <h1 align="center">Images Processing optimization </h1>
</p>

Even with OpenCV in C++ using CUDA, it is still too slow. Therefore, in this repository, I will rewrite an OpenCV library to optimize image processing by using Cuda-C


This repository serves a dual purpose: it provides you with a high-performance image processing library that surpasses OpenCV in C++, while also offering an excellent learning resource for those interested in mastering CUDA-C and parallel image processing.

<p align="center">
 <h1 align="center">How to use my code </h1>
</p>
There will be 2 files: src and my lib 
- src: is for learners who want to understand how the image processing library and CUDA-C are implemented.
- my lib: is for users who just need to substitute OpenCV with my library 


<p align="center">
 <h1 align="center">Requirements </h1>
</p>

- OpenCV C++ (with Cuda)
- Cuda

To check if you have CUDA installed, try running the following command:
`nvcc -V`

![image](https://github.com/CisMine/Cuda-image-processing/assets/122800932/a2a76a50-207c-4bb4-bdfd-2a8546bf452f)


To check if you have OpenCV C++ (with Cuda) installed, try running [this code](https://github.com/CisMine/Cuda-image-processing/blob/main/Check_Opencv.cpp)
```
g++ Check_Opencv.cpp  -std=c++11 `pkg-config --cflags --libs opencv`
./a.out
```

![image](https://github.com/CisMine/Cuda-image-processing/assets/122800932/f1278095-8e82-4036-9adb-fd4b3cbe3025)


<p align="center">
 <h1 align="center"> Installation </h1>
</p>

Flowing [these instructions](https://github.com/CisMine/Setup-as-Cuda-programmers/blob/main/CuDNN%20-%20OpenCV.md) for setting up







<p align="center">
 <h1 align="center"> </h1>
</p>

















