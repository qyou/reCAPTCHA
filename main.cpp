#include "librecaptcha2.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>


void help(void) {
    printf("Copyright (c) NLPR 2015 by You Qiang <qyou@nlpr.ia.ac.cn>\n");
    printf("Usage: librecaptcha2_main ImageFilePath\n\n");
}

int main(int argc, char *argv[]) {
    //test_net();
    char *filepath = NULL;
    if (argc < 2) {
        help();
        return -1;
    } else {
        filepath = argv[1];
    }
    char ret[5] ="\0";
    char formula[6] = "\0";
    double low_threshold_val = 160.0; // Important, the smaller, the less the noise
    double high_threshold_val = 200.0 ; // Important, the larger, the more the information but with more noise

    // read from filepath
    //recaptcha2(filepath, formula, ret, low_threshold_val, high_threshold_val);
    //printf("%s, %s\n", formula, ret);

    struct stat buf_stat;
    int status = stat(filepath, &buf_stat);
    if (0 == status) {
        char *buf = (char *)malloc(buf_stat.st_size * sizeof(char));
        if (buf) {
            FILE *fp = fopen(filepath, "rb");
            if (fp) {
                fread(buf, sizeof(char), buf_stat.st_size, fp);
                fclose(fp);
                recaptcha2_from_buf(buf, buf_stat.st_size, formula, ret, low_threshold_val, high_threshold_val);
                printf("%s, %s\n", formula, ret);
                // Or you can simplly use the following code
                // simple_recaptcha2_from_buf(buf, buf_stat.st_size, ret);
                // printf("%s\n", ret);
                //int result = get_recaptcha2_result_from_buf(buf, buf_stat.st_size);
                //printf("%d\n", result);
            }
            free(buf);
        }
    }
    return 0;
}
