#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#define BYTESPERPIXEL 3

typedef struct {
    uint16_t bfType;      // File type ("BM")
    uint32_t bfSize;      // Size of the file (in bytes)
    uint16_t bfReserved1; // Reserved (must be 0)
    uint16_t bfReserved2; // Reserved (must be 0)
    uint32_t bfOffBits;   // Offset to start of pixel data
} bm_file_header_t;

typedef struct {
    uint32_t biSize;          // Size of the header (40 bytes)
    int32_t  biWidth;         // Width of the image
    int32_t  biHeight;        // Height of the image
    uint16_t biPlanes;        // Number of color planes (must be 1)
    uint16_t biBitCount;      // Bits per pixel (e.g., 24)
    uint32_t biCompression;   // Compression type (0 = none)
    uint32_t biSizeImage;     // Image size (can be 0 for uncompressed)
    int32_t  biXPelsPerMeter; // Pixels per meter (X)
    int32_t  biYPelsPerMeter; // Pixels per meter (Y)
    uint32_t biClrUsed;       // Number of colors
    uint32_t biClrImportant;  // Important colors
} bm_info_header_t;

typedef struct {
    float r,g,b;
} rgb_t;

typedef struct {
    int startY;
    int endY;
    rgb_t *pixels;
    uint8_t *outPixelData;
    int width;
    float avgLum;
    float exposureKey;
    uint8_t *pixelData;
    int pixelDataSize;
} thread_data_t;




float dot(rgb_t a, rgb_t b) {
    return a.r * b.r + a.g * b.g + a.b * b.b;
}

//Function to compute the log average luminance
float computeLogAvgLuminance(rgb_t *pixels, int pixelcount) {
    float logSum = 0.0;
    float delta = 1e-4;
    
    // Luminance weights for RGB to grayscale conversion
    rgb_t luminanceWeights = {0.2126f, 0.7152f, 0.0722f};

    for (int i = 0; i < pixelcount; i++) {
        // Calculate luminance using the dot product of the pixel and the luminance weights
        float L = dot(pixels[i], luminanceWeights);
        logSum += log(delta + L);
    }
    
    // Return the exponent of the average log luminance
    return exp(logSum / (float)pixelcount);
}

rgb_t multiplyScalar(rgb_t color, float scalar) {
    color.r *= scalar;
    color.g *= scalar;
    color.b *= scalar;
    return color;
}


//do this for each color: R,G and B
//a is the exposure key
// avgLum needs to be pre-calculated
rgb_t toneMapReinhard(rgb_t color, float avgLum, float a) {
    // Convert to luminance (perceived brightness)

    // Avoid dividing by zero

    
    
    rgb_t luminanceWeights = {0.2126f, 0.7152f, 0.0722f};

    float L = dot(color, luminanceWeights); // Calculate luminance

   
    // Scale luminance based on exposure key
    float L_scaled = (a / avgLum) * L;

    

    //printf("L scaled: %f\n", L_scaled);

    // Reinhard tone mapping
    float L_mapped = L_scaled / (1.0f + L_scaled);

    

    if (L < 1e-5) {
        return (rgb_t){0.0f, 0.0f, 0.0f}; // Return black if luminance is too low
    }

    // Reapply color ratio (preserve chrominance)
    float scale = L_mapped / L;
    //printf("scale: %f\n", scale);
    return multiplyScalar(color, scale);
}

void *process_pixels(void *arg) {
    thread_data_t *data = (thread_data_t *)arg;
    for (int y = data->startY; y < data->endY; y++) {
        for (int x = 0; x < data->width; x++) {

            int index = x + y * data->width; //index by pixel, not bytes
            rgb_t original = data->pixels[index];
            rgb_t mapped = toneMapReinhard(original, data->avgLum, data->exposureKey);
            int byteIndex = 3 * x + 3 * y * data->width;

            // Clamp the value between 0 and 255
            data->outPixelData[byteIndex]     = (uint8_t)(fminf(fmaxf(mapped.b * 255.0f, 0.0f), 255.0f));
            data->outPixelData[byteIndex + 1] = (uint8_t)(fminf(fmaxf(mapped.g * 255.0f, 0.0f), 255.0f));
            data->outPixelData[byteIndex + 2] = (uint8_t)(fminf(fmaxf(mapped.r * 255.0f, 0.0f), 255.0f));
        }
    }
    return NULL;
}



// Thread function for reading pixel data
void *read_pixel_data(void *arg) {
    thread_data_t *data = (thread_data_t *)arg;
    int startY = data->startY;
    int endY = data->endY;
    int width = data->width;
    uint8_t *pixelData = data->pixelData;
    rgb_t *pixels = data->pixels;
    
    for (int y = startY; y < endY; y++) {
        for (int x = 0; x < width; x++) {
            int index = (y * width + x) * 3; // Compute the starting index for the pixel
            rgb_t pixel;
            pixel.b = pixelData[index] / 255.0f;
            pixel.g = pixelData[index + 1] / 255.0f;
            pixel.r = pixelData[index + 2] / 255.0f;
            
            // Store the pixel data in the corresponding pixel array location
            pixels[x + y * width] = pixel;
        }
    }
    return NULL;
}





int main(int argc, char **argv){

    if (argc != 5){
        printf("usage: %s input.bmp output.bmp exposure_key num_threads\n", argv[0]);
        exit(1);
    }
    double exposure_key = atof(argv[3]);
    int num_threads = atoi(argv[4]);
    if (num_threads == 0){
        printf("num threads cannot be zero\n");
        exit(1);
    }
    
    FILE *f = fopen(argv[1], "rb");
    if (f == NULL) {
        perror("Failed to open file");
        return 1;
    }
    FILE *out = fopen(argv[2], "wb");
    if (out == NULL) {
        perror("Failed to open output file");
        return 1;
    }
    
    bm_file_header_t file_header;
    bm_info_header_t info_header;

    fread(&file_header.bfType, sizeof(uint16_t), 1, f);
    fread(&file_header.bfSize, sizeof(uint32_t), 1, f);
    fread(&file_header.bfReserved1, sizeof(uint16_t), 1, f);
    fread(&file_header.bfReserved2, sizeof(uint16_t), 1, f);
    fread(&file_header.bfOffBits, sizeof(uint32_t), 1, f);

    fread(&info_header, sizeof(bm_info_header_t), 1, f);

    if (info_header.biSizeImage == 0) {
        int rowSize = ((info_header.biBitCount * info_header.biWidth + 31) / 32) * 4;
        info_header.biSizeImage = rowSize * abs(info_header.biHeight);
        printf("image size is zero\n");
    }   

    uint8_t *pixelData = malloc(info_header.biSizeImage);
    if (!pixelData){
        printf("failed to allocate memory for pixel data");
        exit(1);
    }

    size_t bytesRead = fread(pixelData, 1, info_header.biSizeImage, f);
    if (bytesRead != info_header.biSizeImage) {
        printf("Failed to read pixel data\n");
        free(pixelData);
        fclose(f);
        exit(1);
    }

    

    uint8_t *outPixelData = malloc(info_header.biSizeImage);
    if (!outPixelData){
        printf("failed to allocate outPixelData");
        exit(1);
    }

    //int width = (info_header.biWidth * BYTESPERPIXEL + 3) & ~3;
    int width = (info_header.biWidth  + 3) & ~3;


    int num_pixels = info_header.biWidth * abs(info_header.biHeight);
   
    rgb_t *pixels = malloc(sizeof(rgb_t) * info_header.biHeight*info_header.biWidth);
    if (!pixels) {
        perror("Failed to allocate pixel array");
        exit(1);
    }
    
    // for (int y = 0; y < info_header.biHeight; y++){
    //     //printf("y: %d\n", y);
    //     for (int x = 0; x < info_header.biWidth; x++){

    //         rgb_t pixel;
    //         pixel.b = pixelData[3*x + 3*y*width] / 255.0f;
    //         pixel.g = pixelData[3*x + 3*y*width + 1] / 255.0f;
    //         pixel.r = pixelData[3*x + 3*y*width + 2] / 255.0f;
            
    //         pixels[x + y * info_header.biWidth] = pixel;
   
    //     }
    // }


    // Create threads to read pixel data

    pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
    thread_data_t *thread_data = malloc(num_threads * sizeof(thread_data_t));
    int rows_per_thread = info_header.biHeight / num_threads;
    int pixelDataSize = info_header.biSizeImage;

    for (int i = 0; i < num_threads; i++) {
        thread_data[i].startY = i * rows_per_thread;
        thread_data[i].endY = (i == num_threads - 1) ? info_header.biHeight : (i + 1) * rows_per_thread;
        thread_data[i].pixels = pixels;
        thread_data[i].pixelData = pixelData;
        thread_data[i].pixelDataSize = pixelDataSize;
        thread_data[i].width = info_header.biWidth;

        pthread_create(&threads[i], NULL, read_pixel_data, &thread_data[i]);
    }

    // Wait for threads to finish reading the data
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    

    

    
    float avgLum = computeLogAvgLuminance(pixels, num_pixels);
    printf("avgLum: %f\n", avgLum);

    
    
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].startY = i * rows_per_thread;
        thread_data[i].endY = (i == num_threads - 1) ? info_header.biHeight : (i + 1) * rows_per_thread;
        thread_data[i].pixels = pixels;
        thread_data[i].outPixelData = outPixelData;
        thread_data[i].width = info_header.biWidth;
        thread_data[i].avgLum = avgLum;
        thread_data[i].exposureKey = exposure_key;

        pthread_create(&threads[i], NULL, process_pixels, &thread_data[i]);
    }

    // Wait for all threads to finish
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    // for (int y = 0; y < info_header.biHeight; y++) {
    //     for (int x = 0; x < info_header.biWidth; x++) {

    //         int index = x + y * info_header.biWidth;  // index into pixels[]
    //         rgb_t original = pixels[index];

    //         // Apply tone mapping
    //         rgb_t mapped = toneMapReinhard(original, avgLum, exposure_key);

            
    //         int pixelIndex = 3 * x + 3 * y * width;
        
    //                                         //clamp the value to betwee 0 and 255
    //         outPixelData[pixelIndex]     = (uint8_t)(fminf(fmaxf(mapped.b * 255.0f, 0.0f), 255.0f));
    //         outPixelData[pixelIndex + 1] = (uint8_t)(fminf(fmaxf(mapped.g * 255.0f, 0.0f), 255.0f));
    //         outPixelData[pixelIndex + 2] = (uint8_t)(fminf(fmaxf(mapped.r * 255.0f, 0.0f), 255.0f));

    //     }
    // }

    fwrite(&file_header.bfType, sizeof(uint16_t), 1, out);
    fwrite(&file_header.bfSize, sizeof(uint32_t), 1, out);
    fwrite(&file_header.bfReserved1, sizeof(uint16_t), 1, out);
    fwrite(&file_header.bfReserved2, sizeof(uint16_t), 1, out);
    fwrite(&file_header.bfOffBits, sizeof(uint32_t), 1, out);

    fwrite(&info_header, sizeof(bm_info_header_t), 1, out);
    fwrite(outPixelData, 1, info_header.biSizeImage, out);
    fclose(out);
    fclose(f);


    //free malloced data
    free(threads);
    free(thread_data);
    free(pixels);
    free(outPixelData);
    free(pixelData);
    


    return 0;
}