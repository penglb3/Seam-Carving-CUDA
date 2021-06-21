# Seam-Carving-CUDA

[TOC]

## 环境

以下的环境是经过测试可以运行的

- Windows 10
- Visual Studio 2019 (vc16)
- OpenCV 4.5.2
- NVCC 11.1

如果你希望使用其他环境，可以参考这个环境的配置自己做。

### 配置OpenCV

#### 下载

从官网下载：https://sourceforge.net/projects/opencvlibrary/files/4.5.2/opencv-4.5.2-vc14_vc15.exe/download

解压到你喜欢的地方，最好路径不含空格和中文。现在假设这个目录（有`README.md.txt`的目录）是`.../opencv`

下载opencv_contrib：https://github.com/opencv/opencv_contrib

拷贝`modules/cudev`文件夹到`.../opencv/src/modules`文件夹里面。

#### CMake配置

在CMake中把源代码目录选择为`.../opencv/sources`，目标二进制文件目录`.../opencv/build/x64/vc16`。因为这个`vc16`不存在，会询问你是否创建，选择是就可以了。

点击左下方`Configure`，等它搞完，**然后在中间的各项配置中找到`WITH_CUDA`并把它勾上。**

点`Generate`。如果这里失败的话，检查`WITH_CUDNN`、`WITH_CUFFT`、`WITH_CUBLAS`三个选项，把它们取消选择，因为我们不需要他们。

（可选：`CUDA_ARCH_BIN`选项中删去所有小于等于`5.0`的版本，这样少一些警告）

#### VS 2019编译

Generate成功后到目标二进制文件目录，应该有一个`OpenCV.sln`，打开它，**先在上面把编译模式设为Release，再点编译**。

编译成功后记得把`.../opencv/build/x64/vc16/bin/Release`加入到你的路径中，后面**运行**程序时会需要里面的dll文件。

## 编译

直接用nvcc把所有的文件一次编译就可以，编译时需要指定一些OpenCV的头文件目录和链接必要的库。

需要：

- 指明Include目录：`-I .../opencv/build/include`（或者链接项目目录下的`include/`也可以）
- 指明链接库目录：`-L .../opencv/build/x64/vc16/lib/Release`（或者链接项目目录下的`lib/`也可以）
- 链接如下库：`-l opencv_core452 -l opencv_imgproc452 -l opencv_imgcodecs452 -l opencv_highgui452 -l opencv_photo452 -l videoio452`

如果不想每次都整这么一大堆参数，我们提供了VS，VSCode，或者CMake的编译和包含设置可供使用。其中前二者的配置文件在`src/`下，后者的配置文件在项目目录。

## 运行

### 参数

- `-w [int]`：当参数为正数时，指明目标图像宽度；当参数为负数时，指明要减少的宽度。默认`0`。
- `-h [int]`：当参数为正数时，指明目标图像高度；当参数为负数时，指明要减少的高度。默认`0`。
- `-i [str]`：指明输入图像的路径，默认是`../images/Tension.jpg`。
- `-o [str]`：指明输出图像的路径，默认是`../images/output.jpg`。
- `-s`：当该参数被声明时，将使用CPU版本。
- `-v`：当该参数被声明时，将进行可视化产生视频。此时计算用时不再可参考。
- `-f [unsigned int]`：该参数声明可视化视频的fps。默认是`30`。

### 注意事项

1. block的大小固定为32x32或1024。
2. 目前程序仅支持图像减小，不支持增大。所以请注意`-w`和`-h`的值。
3. `-w`和`-h`至少一个非0，否则视为不需要缩放，程序直接退出。
4. 默认（即不声明`-s`参数）会使用CUDA版本。

