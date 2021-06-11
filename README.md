# Seam-Carving-CUDA

## 环境

以下的环境是经过测试可以运行的

- Windows 10
- Visual Studio 2019 (vc16)
- OpenCV 4.5.2
- NVCC 11.1
- 编辑器：Visual Studio Code

如果你希望使用其他环境，可以参考这个环境的配置自己做。

### 配置OpenCV

从官网下载：https://sourceforge.net/projects/opencvlibrary/files/4.5.2/opencv-4.5.2-vc14_vc15.exe/download

解压到你喜欢的地方，最好路径不含空格和中文。现在假设这个目录（有`CMakeList.txt`的目录）是`.../opencv`

然后用CMake，把源代码目录选择为`.../opencv/sources`，目标二进制文件目录`.../opencv/build/x64/vc16`。

点击左下方`Configure`，等它搞完，点旁边的`Generate`。

Generate完后到目标二进制文件目录，应该有一个`OpenCV.sln`，打开它，**先在上面把编译模式设为Release，再点编译**。

等编译完之后，把`.../opencv`和`.../opencv/build/x64/vc16`

## 编译

直接用nvcc把所有的文件一次编译就可以，编译时需要指定一些OpenCV的目录和加载必要的动态库。

需要：

- 指明Include目录：`-I .../opencv/build/include`
- 指明链接库目录：`-L .../opencv/build/x64/vc16/lib/Release`
- 链接如下库：`-l opencv_core452 -l opencv_imgproc452 -l opencv_imgcodecs452 -l opencv_highgui452 -l opencv_photo452 `

如果不想每次都整这么一大堆参数，可以用VS，VSCode，或者CMake来配置对应的编译和包含设置。