#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define BYTESPERPIXEL 3


int main(int argc, char **argv){

    int biWidth = atoi(argv[1]);
    int width = (biWidth * BYTESPERPIXEL + 3) & ~3;
    printf("width: %d\n", width);

}