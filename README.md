#reCAPTCHA 开者指南#

---
##简介##
reCAPTCHA是一个识别验证码的程序，目的就是为了按需求解决验证码识别的问题。其识别思路简单清晰，具体用到的技术概要叙述如下：
1. 图片接口的转换模块。通过[FreeImage](http://freeimage.sourceforge.net/ "FreeImage")和[OpenCV](http://opencv.org/ "OpenCV")之间接口的转换，使得可处理的图片格式大大增加，[OpenCV](http://opencv.org/ "OpenCV")本身是处理不了`gif`格式的，但是我们识别的图片都是互联网上流行的`gif`静态图片，需要做转换，转换后，基本识别全部图片格式，具体的图片格式，详见[FreeImage支持的图片格式列表](http://freeimage.sourceforge.net/features.html)。
2. 图片的预处理模块。原始的图片噪声很多、并且文字有很多都有一定程度的倾斜，对后面的识别不利。首先通过[OpenCV](http://opencv.org/ "OpenCV")中图像平滑（主要是中值滤波[medianBlur](http://docs.opencv.org/modules/imgproc/doc/filtering.html?highlight=medianblur#void medianBlur(InputArray src, OutputArray dst, int ksize)和形态学作用[morphologyEx](http://docs.opencv.org/modules/imgproc/doc/filtering.html?highlight=morphologyex#cv2.morphologyEx)）去除噪声，然后使用轮廓发现算法（[findContours](http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#findcontours)）提取单个字符完成字符的切分，逐个完成字符的矫正。最后对矫正后的字符二值化送到后面的识别模块中完成识别。
3. 识别模块。目前reCAPTCHA有两个版本，1和2。其中其中`librecaptcha1`使用的是`Google`的[tesseract](https://code.google.com/p/tesseract-ocr/ "tesseract")；而`librecaptcha2`使用的是[cvconvnet](https://github.com/eldog/fface/tree/master/src/conv-net-0.1)，这是一个很强大的卷积神经网络机器学习算法，广泛应用于语音识别、图像识别之中，我们通过完成数字的训练，可以大大提高准确性。两者各有利弊，前者准确度会低于后者，但是速度要大大快于后者。

##编译代码##
代码使用cmake管理编译，所以你首先需要安装cmake

```bash
sudo apt-get install cmake
```

然后进入对应的版本目录，仔细阅读`COMPILE.txt`文档，按流程完成编译操作，每一个版本下都有对应已经完成的编译版本在`release`目录下，编译平台`Ubuntu 12.04 LTS X64`，可以直接使用。

##头文件接口##
+ 版本`librecaptcha1`中的头文件`recaptcha.h`说明如下
```c
char *recaptcha(const char *filepath, char *formula, char *result, double low_thres_val, double high_thres_val);
功能：输入图像文件的地址，通过双阈值`low_thres_val`和`high_thres_val`，输入图片识别的字符串`formula`，以及计算的结果`result`
注意：低阈值`low_thres_val`最佳范围在140.0到190.0，推荐160,高阈值`low_thres_val`最佳范围在190.0到210.0之间，推荐200.0
```
```c
char *recaptcha_from_buf(void *buffer, int buf_size, char *formula, char *result, double low_thres_val, double high_thres_val);
功能：输入图像的内存指针`buffer`和大小`buf_size`,得倒对应的结果。其他参数同`recaptcha`
```
```c
char *simple_recaptcha(const char *filepath, char *result);
功能：对`recaptcha`的简化，其中高低阈值使用的是推荐值，只返回计算得到的结果，以char*保存。
```
```c
char *simple_recaptcha_from_buf(void *buf, int buf_size, char *result);
功能：对`recaptcha_from_buf`的简化
```
```c
int get_recaptcha_result(const char *filepath);
功能：对`recaptcha`的简化，其中高低阈值使用的是推荐值，只返回计算得到的结果，以int保存。
```
```c
int get_recaptcha_result_from_buf(void *buf, int buf_size);
```

+ 版本`librecaptcha2`中的头文件`librecatpcha2.h`只是将所用`recaptcha`的地方替换为`recaptcha2`,其他使用方法同`librecatpcha1`


##模块概要##
+ ###图片接口的转换模块###
主要函数`GenericLoader`将图片载入FreeImage对应的内存格式中，`do_trans`完成FreeImage对应内存格式到OpenCV格式的转换，`do_load`完成最终的图片载入工作。将上述过程打包操作的函数为`load_to_mat`和`load_to_mat_from_mem`，具体的函数原型
```c
int load_to_mat(const char *filepath, Mat &mat)
```
```c
int load_to_mat_from_mem(void *buffer, int buf_size, Mat &mat)
```


+ ###图片的预处理模块###
主要函数`smooth_morphology`完成图片的平滑和形态学操作，`threshold_and_contours`完成初期二值化和轮廓提取,
`affine_contour`和`refine_contour`完成图像的矫正和精细化操作，`get_main_elements`完成图像字符的切分和轮廓提取。将上述函数打包的最终函数为`get_adaptive_elements`，完成符号和数字的提取。其函数原型为
```c
int get_adaptive_elements(Mat &data, double low_thres_val, double high_thres_val, vector<Point> &cnt1, vector<Point> &cnt2, vector<Point> &cnt_mark)
```

+ ###识别模块###
由于版本1和2采用了不同的方法，故分别加以介绍:
版本1使用的核心函数为`ocr_read_simple`和`do_recatpcha`，其函数原型为：
```c
char *ocr_read_simple(tesseract::TessBaseAPI *api, const Mat &org, vector<Point> &contour, Mat &ret)
```
```c
char *do_recaptcha(const Mat &img, char *formula, char *result, double low_thres_val, double high_thres_val)
```
版本2使用的核心函数为`net_read_simple`和`do_recatpcha2`，其函数原型为：
```c
int net_read_simple(CvConvNet *pNet, const Mat &org, vector<Point> contour, Mat &ret)
```
```c
char* do_recaptcha2(const Mat &img,  char *formula, char *result, double low_thres_val, double high_thres_val)
```



 



