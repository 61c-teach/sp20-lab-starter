
#ifndef COMPUTE_DOTP_H
#define COMPUTE_DOTP_H

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "libbmp/libbmp.h"

// #define ARRAY_SIZE 100000
// extern int ARRAY_SIZE;
// #define ARRAY_SIZE 100000000
// #define REPEAT     100
#define REPEAT 10
#define BUF_SIZE 8192

char* compute_dot(int);
char *image_proc(const char*);

#endif