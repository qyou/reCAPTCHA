#ifndef MNIST_SVM_MODEL_LIBRECAPTCHA2_H
#define MNIST_SVM_MODEL_LIBRECAPTCHA2_H

/*
 * filepath: the image file path, because we use FreeImage library, so many image formats are supported
 * formula: the chars in the image
 * result: the return result
 * low_thres_val, high_thres_val: the two threshold of the image. the larger, the more noise but more information may get.
 * YOU SHOULD SET low_thres_val=140~190.0 (160.0 preferred)  and high_thres_val=190~210.0 (200 prefered)in our digit recognition problem
 */
char *recaptcha2(const char *filepath, char *formula, char *result, double low_thres_val, double high_thres_val);
/*
 * buffer and buf_size replace the filepath
 */
char *recaptcha2_from_buf(void *buffer, int buf_size, char *formula, char *result, double low_thres_val, double high_thres_val);

/*
 * simpler from recaptcha where we set low_thres_val=160.0 and high_thres_val=200.0 and ignore the formula in the image
 * and return the final result
 */
char *simple_recaptcha2(const char *filepath, char *result);
/*
 * buffer and buf_size replace the filepath
 */
char *simple_recaptcha2_from_buf(void *buffer, int buf_size, char *result);

/*
 * more precise from simple_recaptcha, only get the final result from the image and parse it to int
 */
int get_recaptcha2_result(const char *filepath);
int get_recaptcha2_result_from_buf(void *buffer, int buf_size);

#endif //MNIST_SVM_MODEL_LIBRECAPTCHA2_H
