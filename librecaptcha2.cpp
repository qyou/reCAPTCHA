#include "librecaptcha2.h"

#include <iostream>
#include <fstream>
#include <sstream>

#include <FreeImage.h>

#include <opencv2/opencv.hpp>
#include <cvconvnet.h>
using namespace cv;
using namespace std;

//通用图像加载函数，支持的图像有bmp,jpg,tif,png,gif,psd,pgm等等
static FIBITMAP* GenericLoader(const char* filename,int flag)
{

    FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;
    fif = FreeImage_GetFileType(filename,0);//获取文件的类型标签
    if(fif == FIF_UNKNOWN)//如果文件没有类型标签
    {
        fif = FreeImage_GetFIFFromFilename(filename);//从文件名的后缀猜测文件类型
    }
//文件被该库支持
    if(fif != FIF_UNKNOWN && FreeImage_FIFSupportsReading(fif))
    {
        FIBITMAP* dib = FreeImage_Load(fif,filename,flag);
        return dib;
    }
    return NULL;
}

static FIBITMAP* GnericLoaderFromMem(void *buffer, int buf_size, int flag)
{
    if (buffer == NULL && buf_size <= 0) {
        fprintf(stderr, "Empty Image Buffer!\n");
        return NULL;
    }

    FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;
    FIMEMORY *hmem = FreeImage_OpenMemory((BYTE*) buffer, buf_size);
    fif = FreeImage_GetFileTypeFromMemory (hmem, 0);
    if (fif != FIF_UNKNOWN && FreeImage_FIFSupportsReading(fif)) {
        FIBITMAP *dib = FreeImage_LoadFromMemory(fif, hmem, flag);
        FreeImage_CloseMemory(hmem);
        return dib;
    }
    FreeImage_CloseMemory(hmem);
    return NULL;
}

static IplImage *do_trans(FIBITMAP *dib) {
    int nClrUsed = FreeImage_GetColorsUsed(dib);
    int height = FreeImage_GetHeight(dib);
    int width = FreeImage_GetWidth(dib);
    RGBQUAD* pPalette = FreeImage_GetPalette(dib);
    int nChannel=3;
    if(!nClrUsed && !pPalette)		//无调色板图像处理
    {

        IplImage* iplImg = cvCreateImage(cvSize(width,height),IPL_DEPTH_8U,nChannel);
        iplImg->origin = 1;
        for(int y=0;y<height;y++)
        {
            BYTE* pLine = (BYTE*)iplImg->imageData + y*iplImg->widthStep;
            BYTE* psrcLine = (BYTE*)FreeImage_GetScanLine(dib,y);
            for (int x=0;x<nChannel*width;x++)
            {
                *pLine++ = *psrcLine++;

            }
        }
        return iplImg;
    }
    else if(pPalette)//索引图像处理
    {
        IplImage* iplImg = cvCreateImage(cvSize(width,height),IPL_DEPTH_8U,nChannel);
        iplImg->origin = 1;
        BYTE intensity;
        BYTE* pIntensity = &intensity;
        for(int y=0;y<height;y++)
        {
            BYTE* pLine = (BYTE*)iplImg->imageData + y*iplImg->widthStep;
            for (int x=0;x<width;x++)
            {

                FreeImage_GetPixelIndex(dib,x,y,pIntensity);
                pLine[x*3] = pPalette[intensity].rgbBlue;
                pLine[x*3+1] = pPalette[intensity].rgbGreen;
                pLine[x*3+2] = pPalette[intensity].rgbRed;
            }
        }
        return iplImg;
    }
    else
    {
        return NULL;
    }
}

static IplImage* pic2ipl(const char* filename)
{
    FreeImage_Initialise();
    FIBITMAP* dib = GenericLoader(filename, 0);
    if(!dib)
        return NULL;
    IplImage *result = do_trans(dib);
    FreeImage_Unload(dib);
    dib = NULL;
    FreeImage_DeInitialise();
    return result;

}

static IplImage* pic2ipl_from_mem(void *buffer, int buf_size)
{
    FreeImage_Initialise();
    FIBITMAP* dib = GnericLoaderFromMem(buffer, buf_size, 0);
    if(!dib)
        return NULL;
    IplImage *result = do_trans(dib);
    FreeImage_Unload(dib);
    dib = NULL;
    FreeImage_DeInitialise();
    return result;
}

static IplImage *flip(IplImage *data) {
    IplImage *gray = cvCreateImage(cvGetSize(data), data->depth, 1);
    cvCvtColor(data, gray, CV_BGR2GRAY);
    IplImage *flipImage = cvCreateImage(cvGetSize(data), data->depth, 1);
    cvFlip(gray, flipImage, 0);
    cvReleaseImage(&gray);
    gray = NULL;
    cvReleaseImage(&data);
    data = NULL;
    return flipImage;
}

static IplImage *load_to_ipl(const char *filepath) {
    IplImage *data = pic2ipl(filepath);
    if (NULL == data) {
        fprintf(stderr, "connot load the image %s\n", filepath);
        return NULL;
    }

    return flip(data);
}

static IplImage *load_to_ipl_from_mem(void *buffer, int buf_size) {
    IplImage *data = pic2ipl_from_mem(buffer, buf_size);
    if (NULL == data) {
        fprintf(stderr, "connot load the image.\n");
        return NULL;
    }

    return flip(data);
}



static int ipl_to_mat(IplImage *flip_data, Mat &mat) {
    mat = cvarrToMat(flip_data, true);
    cvReleaseImage(&flip_data);
    return 0;
}


static int load_to_mat(const char *filepath, Mat &m) {
    IplImage *ipl = load_to_ipl(filepath);
    if (!ipl) {
        return -1;
    }
    return ipl_to_mat(ipl, m);
}

static int load_to_mat_from_mem(void *buffer, int buf_size, Mat &mat) {
    IplImage *ipl = load_to_ipl_from_mem(buffer, buf_size);
    if (NULL == ipl) {
        fprintf(stderr, "connot load the image from buffer!\n");
        return -1;
    }

    return ipl_to_mat(ipl, mat);
}

static bool compare_size_func(const vector<Point> &c1, const vector<Point> &c2) {
    return (c1.size() > c2.size());
}
/*
 * JUST for Test, not used in release version
 */
/*static bool compare_x_func(const vector<Point> &c1, const vector<Point> &c2) {
    Rect rt1 = boundingRect(c1);
    Rect rt2 = boundingRect(c2);
    return (rt1.x < rt2.x);

} //*/
static bool compare_y_func(const Point2f &pt1, const Point2f &pt2) {
    return (pt1.y < pt2.y);
}

static bool is_right_lean(const RotatedRect &rotated_rect)
{
    Point2f vertices[4];
    rotated_rect.points(vertices);
    vector<Point2f> points;
    for(int i=0; i<4; ++i) {
        points.push_back(vertices[0]);
    }
    sort(points.begin(), points.end(), compare_y_func);

    return points[0].x  + points[1].x > 2 * rotated_rect.center.x;

}

/*
 * JUST FOR Test, not used in release version
 *

static void show(const char *window_name, const Mat &image) {
    namedWindow(window_name, WINDOW_AUTOSIZE);
    imshow(window_name, image);
}

//*/

static bool is_plus(const vector<Point> &contour) {
    Rect rt = boundingRect(contour);
//    RotatedRect rotatedRect = minAreaRect(contour);
//    vector<double> tmp;
//
//    for (int i=0; i<contour.size(); ++i) {
//        double x = (abs(contour[i].x-rotatedRect.center.x));
//        tmp.push_back(x);
//    }
//    sort(tmp.begin(), tmp.end());
    //TODO: Need to improved the algorithm
    return (rt.height > 2 && rt.width < 1.4*rt.height); // only an empirical result, should be improved
}

static void shift_mat(const Mat &mat, Mat &ret) {
    warpAffine(ret, ret, mat, Size(ret.cols, ret.rows), INTER_LINEAR, BORDER_CONSTANT, 255);

}

static void shift_center(vector<Point> &contour, Mat &ret) {
    if (contour.size() > 5) {
        RotatedRect rt = fitEllipse(contour);
        double sx = ret.cols / 2 - rt.center.x;
        double sy = ret.rows / 2 - rt.center.y;
        Mat m = (Mat_<double>(2, 3) << 1.0, 0.0, sx, 0.0, 1.0, sy);
        warpAffine(ret, ret, m, Size(ret.cols, ret.rows));
    }
}

static Mat normalize_mat(vector<Point> &contour, const Mat &data) {
    Rect rt = boundingRect(contour);
    int unisize = max(rt.width, rt.height);
    int new_x = rt.x + rt.width/2-unisize/2;
    int new_y = rt.y + rt.height/2-unisize/2;
    Rect newRect(new_x, new_y, unisize, unisize);
    Mat tmp1 =  data(newRect).clone();
    int MNIST_SIZE = 28;
    Mat tmp2 = Mat::zeros(Size(MNIST_SIZE, MNIST_SIZE), CV_MAKETYPE(data.depth(), 1));
    resize(tmp1, tmp2, tmp2.size(), 0, 0, CV_INTER_LINEAR);
    return tmp2;
}

static Mat padding_to_32(const Mat &mat_28) {
    int TEST_SIZE = 32;
    Mat tmp2 = Mat::zeros(Size(TEST_SIZE, TEST_SIZE), CV_MAKETYPE(mat_28.depth(), 1));
    copyMakeBorder(mat_28, tmp2, 2,2,2,2,BORDER_CONSTANT, Scalar(0,0,0));
    return tmp2;
}

/**
 * smooth and morphology to remove much moise
 */

static void smooth_morphology(const Mat &image, Mat &dst, int blur_size, int morpholy_size) {


    Mat smooth_image(image.size(), CV_MAKETYPE(image.depth(), 1));
    Mat kernel=Mat::ones( morpholy_size, morpholy_size, CV_MAKETYPE(image.depth(), 1))/ (float)(morpholy_size*morpholy_size);
    medianBlur(image, smooth_image, blur_size);
    morphologyEx(smooth_image, dst, MORPH_OPEN, kernel);
}


/**
 * threshod the image, and then get the contours to extract the main partial elements
 */

static int threshold_and_contours(const Mat &morph, double thres, double max, Mat &dst, vector<vector<Point> > &contours) {
    threshold(morph, dst, thres, max, THRESH_BINARY_INV);

    Mat tmp = dst.clone();
    vector<Vec4i> hierarchy;

    findContours(tmp, contours, hierarchy,
                 CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    if (contours.size() < 2) {
        fprintf(stderr, "Not so many elements! Please increase the threshold value!\n");
        return -1;
    }

    sort(contours.begin(), contours.end(), compare_size_func);
    return 0;
}

/**
 * rotate the image to make it straight
 */
static int affine_contour(vector<Point> &cnt, const Mat &thresh,  Mat &dst) {
    Mat mask = Mat::zeros(thresh.size(), CV_MAKETYPE(thresh.depth(), 1));
    Mat tmp = Mat::zeros(thresh.size(), CV_MAKETYPE(thresh.depth(), 1));

    Rect rt = boundingRect(cnt);
    for (int i=rt.x; i<=rt.x + rt.width; ++i) {
        for (int j=rt.y; j<=rt.y+rt.height; ++j) {
            mask.at<uchar>(j,i) = 255;
        }
    }
    bitwise_and(thresh, thresh, tmp, mask);
    Mat data = tmp.clone();
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours( data, contours, hierarchy,
                  CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE );
    if (contours.empty()) {
        return -1;
    }
    sort(contours.begin(), contours.end(), compare_size_func);
    cnt = contours[0];
    if (cnt.size() > 5) {
        RotatedRect rotated_rect = fitEllipse(cnt);
        double angle = rotated_rect.angle;

        // test whether the image left-leaning or right-leaning
        // the processing strategies are different
        if (is_right_lean(rotated_rect)) {
        } else {
            angle -= 180;
        }

        Mat rotationMatrix = getRotationMatrix2D(rotated_rect.center, angle, 1.0);
        warpAffine(tmp, dst, rotationMatrix, tmp.size());
    } else {
        dst = tmp.clone();
    }

    return 0;
}

/**
 * remove the small partial elements and get the main content
 */

static int refine_contour(vector<Point> &cnt, const Mat &src, Mat &dst) {
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    Mat data = src.clone();
    findContours( data, contours, hierarchy,
                  CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE );
    if (contours.empty()) {
        return -1;
    }
    sort(contours.begin(), contours.end(), compare_size_func);
    cnt = contours[0];

    Mat dst1 = Mat::zeros(src.size(), CV_MAKETYPE(src.depth(), 1));

    Mat mask = Mat::zeros(src.size(), CV_MAKETYPE(src.depth(), 1));

    Rect rt = boundingRect(cnt);
    for (int i=rt.x; i<=rt.x + rt.width; ++i) {
        for (int j=rt.y; j<=rt.y + rt.height; ++j) {
            mask.at<uchar>(j,i) = 255;
        }
    }

    bitwise_and(src, src, dst1, mask);
    if (cnt.size() > 5) {
        RotatedRect rotated_rect = fitEllipse(cnt);
        double angle = rotated_rect.angle;

        if (is_right_lean(rotated_rect)) {

        } else {
            angle -= 180;
        }

        Mat rotationMatrix1 = getRotationMatrix2D(rotated_rect.center, angle, 1.0);
        warpAffine(dst1, dst, rotationMatrix1, dst1.size());
    }  else {
        dst = dst1.clone();
    }
    return 0;
    /*Rect rt2 = boundingRect(cnt);
    double ang=0.0;
    if (rt2.width > rt2.height) {
        ang = 90.0;

    }
    Mat rotationMatrix2 = getRotationMatrix2D(rotated_rect.center, ang, 1.0);
    warpAffine(dst2, dst, rotationMatrix2, dst2.size());//*/
}

static double average_x(const vector<Point> points) {
    int size = points.size();
    double sum = 0.0;
    for (int i=0; i<size; ++i) {
        sum += points[i].x;
    }
    return sum/size;
}

static int get_main_elements(const vector<vector<Point> > &contours, vector<Point> &cnt1, vector<Point> &cnt2) {
    if (contours.size() < 2) {
        fprintf(stderr, "not so many elements, please increase the thresh value!\n");
        return -1;
    }

    vector<Point> v1 = contours[0];
    vector<Point> v2 = contours[1];

    RotatedRect rt1 = minAreaRect(v1);
    RotatedRect rt2 = minAreaRect(v2);
    if (rt1.center.x > rt2.center.x) {
        cnt1 = v2;
        cnt2 = v1;
    } else {
        cnt1 = v1;
        cnt2 = v2;
    }
    return 0;
}

static int get_elements(const vector<vector<Point> > &contours, vector<Point> &cnt1, vector<Point> &cnt2, vector<Point> &cnt_mark)
{
    if (contours.size() < 2) {
        fprintf(stderr, "not so many elements! Please increase the thresh value!\n");
        return -1;
    }

    vector<Point> v1 = contours[0];
    vector<Point> v2 = contours[1];


    RotatedRect rt1 = minAreaRect(v1);
    RotatedRect rt2 = minAreaRect(v2);
    if (rt1.center.x > rt2.center.x) {
        cnt1 = v2;
        cnt2 = v1;
    } else {
        cnt1 = v1;
        cnt2 = v2;
    }

    vector<vector<Point> > sub_contours;
    for (int i=2; i<contours.size(); ++i) {
        double mark_x_center = average_x(contours[i]);
        if (mark_x_center> min(rt1.center.x, rt2.center.x)
            && mark_x_center < max(rt1.center.x, rt2.center.x)) {
            sub_contours.push_back(contours[i]);
        }
    }
    if (sub_contours.empty()) {
        return -2;
    }
    sort(sub_contours.begin(), sub_contours.end(), compare_size_func);
    cnt_mark = sub_contours[0];
    return 0;
}

static char get_mark(const vector<Point> &cnt) {
    char mark;
    if (is_plus(cnt)) {
        mark = '+';
    } else {
        mark = '-';
    }
    return mark;
}

static int get_adaptive_elements(Mat &data, double low_thres_val, double high_thres_val, vector<Point> &cnt1, vector<Point> &cnt2, vector<Point> &cnt_mark)
{
    // In some old machine, the next two line should be commented to avoid segmentation fault
    Mat m = (Mat_<double>(2,3) << 1.0,  0.0, 10.0, 0.0, 1.0, 0.0);
    shift_mat(m, data);

    Mat intial = data.clone();
    Mat tmp(data.size(), CV_MAKETYPE(data.depth(), 1));
    vector<vector<Point> > full_contours;
    vector<vector<Point> > partial_contours;
    //vector<Point>  cnt1_candidate1, cnt2_candidate1;
    vector<Point>  cnt1_candidate2, cnt2_candidate2, cnt_mark_candiate;
    int ret1, ret2;
    int tmp1 = low_thres_val, tmp2 = high_thres_val;

    threshold_and_contours(intial, tmp1, 255, data, partial_contours);
    ret1= get_main_elements(partial_contours, cnt1, cnt2);
    //printf("%ld, %ld\n", cnt1_candidate1.size(), cnt2_candidate1.size());



    threshold_and_contours(intial, tmp2, 255, data, full_contours);
    ret2= get_elements(full_contours, cnt1_candidate2, cnt2_candidate2, cnt_mark_candiate);
    cnt1_candidate2.clear();
    cnt2_candidate2.clear();
    for (vector<vector<Point> >::iterator iter = partial_contours.begin(); iter != partial_contours.end(); ++iter) {
        iter->clear();
    }

    partial_contours.clear();

    for (vector<vector<Point> >::iterator iter = full_contours.begin(); iter != full_contours.end(); ++iter) {
        iter->clear();
    }
    full_contours.clear();

    if (cnt1.empty() || cnt2.empty()) {
        return -1;
    }



    if (ret2 == -2) {
        if (!cnt_mark.empty()) {
            cnt_mark.clear();
        }
        return -2;
    }
    cnt_mark = cnt_mark_candiate;
    return 0;

}


static int net_recognize(CvConvNet *pNet, const Mat &element) {
    IplImage copy = element;
    int r = -1;
    try {
        r = (int)pNet->fprop(&copy);
    }
    catch (exception &e)
    {
        cerr << "Exception: " << e.what() << endl;
    }
    return r;
}

static int net_read_simple(CvConvNet *pNet, const Mat &org, vector<Point> contour, Mat &ret) {
    Mat dst1(org.size(), CV_MAKETYPE(org.depth(), 1));
    Mat dst2(org.size(), CV_MAKETYPE(org.depth(), 1));
    Mat dst3(org.size(), CV_MAKETYPE(org.depth(), 1));

    affine_contour(contour, org, dst1);
    refine_contour(contour, dst1, dst2);
    affine_contour(contour, dst2, dst3);
    refine_contour(contour, dst3, ret);

    //shift_center(contour, ret);
    ret = normalize_mat(contour, ret);
    ret = padding_to_32(ret);
    return net_recognize(pNet, ret);
}



static char* do_recaptcha2(const Mat &img,  char *formula, char *result, double low_thres_val, double high_thres_val) {

    CvConvNet net;

    ifstream ifs("/usr/local/share/conv-net/data/mnist.xml");
    string xml((istreambuf_iterator<char>(ifs)), istreambuf_iterator<char>());
    if (!net.fromString(xml)) {
        cerr << "ERROR: cannot load net from xml" << endl;
        formula = result = NULL;
        return result;
    }

    Mat morph(img.size(), CV_MAKETYPE(img.depth(), 1));
    smooth_morphology(img, morph, 3, 3);

    vector<Point>  cnt1, cnt2, cnt_mark;
    get_adaptive_elements(morph, low_thres_val, high_thres_val, cnt1, cnt2, cnt_mark);

    Mat first(img.size(), CV_MAKETYPE(img.depth(), 1));
    Mat second(img.size(), CV_MAKETYPE(img.depth(), 1));


    int f=-1, s=-1;

    f = net_read_simple(&net, morph, cnt1, first);
    cnt1.clear();
    s = net_read_simple(&net, morph, cnt2, second);
    cnt2.clear();

    char mark_char= cnt_mark.empty()? '-': get_mark(cnt_mark);
    cnt_mark.clear();
    int tmp = 0;
    if (mark_char=='+') {
        tmp = f + s;
    } else {
        tmp = f - s;
    }
    if (tmp>=0) {
        sprintf(result, "%d", tmp);
    } else {
        sprintf(result, "-%d", -tmp);
    }
    sprintf(formula, "%d%c%d=?", f, mark_char, s);
    return result;
}

char *recaptcha2(const char *filepath, char *formula, char *result, double low_thres_val, double high_thres_val) {
    Mat img;
    //image = imread(filepath, IMREAD_GRAYSCALE);
    load_to_mat(filepath, img); // Support multi-format images by using FreeImage library
    if (!img.data) {
        printf("No image data \n");
        formula = result = NULL;
        return result;
    }

    return do_recaptcha2(img, formula, result, low_thres_val, high_thres_val);
}

char *recaptcha2_from_buf(void *buffer, int buf_size, char *formula, char *result, double low_thres_val, double high_thres_val)
{
    Mat img;
    //image = imread(filepath, IMREAD_GRAYSCALE);
    load_to_mat_from_mem(buffer, buf_size, img); // Support multi-format images by using FreeImage library
    if (!img.data) {
        printf("No image data \n");
        formula = result = NULL;
        return result;
    }
    return do_recaptcha2(img, formula, result, low_thres_val, high_thres_val);
}

char *simple_recaptcha2(const char *filepath, char *result) {
    char formula[5] = "\0";
    return recaptcha2(filepath, formula, result, 160, 200);
}

char *simple_recaptcha2_from_buf(void *buffer, int buf_size, char *result) {
    char formula[5] = "\0";
    return recaptcha2_from_buf(buffer, buf_size, formula, result, 160.0, 200.0);
}


int get_recaptcha2_result(const char *filepath) {
    char data[3] = "\0";
    simple_recaptcha2(filepath, data);
    return atoi(data);
}

int get_recaptcha2_result_from_buf(void *buffer, int buf_size) {
    char data[3] = "\0";
    simple_recaptcha2_from_buf(buffer, buf_size, data);
    return atoi(data);
}


